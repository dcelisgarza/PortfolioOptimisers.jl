"""
    abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end

Abstract supertype for all moment-based risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using statistical moments (e.g., mean, variance, skewness, kurtosis) of the return distribution. All concrete moment risk measure algorithms should subtype `MomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
"""
abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end
"""
    abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end

Abstract supertype for all low-order moment-based risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments (e.g., mean, variance, mean absolute deviation) of the return distribution. All concrete low-order moment risk measure algorithms should subtype `LowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`StandardisedLowOrderMoment`](@ref)
"""
abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
    abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end

Abstract supertype for low-order moment risk measure algorithms that are not standardised by the variance in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments without normalising by the variance. All concrete unstandardised low-order moment risk measure algorithms should subtype `UnstandardisedLowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
"""
abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end
function factory(alg::MomentMeasureAlgorithm, args...; kwargs...)
    return alg
end
"""
    struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end

Represents the first lower moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the first lower moment, which is the negative mean of the deviations of the returns series below a target value.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
    struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end

Represents the mean absolute deviation risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk as the mean of the absolute deviations of the returns series from a target value.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
    abstract type UnstandardisedSecondMomentAlgorithm <: UnstandardisedLowOrderMomentMeasureAlgorithm end

Abstract supertype for unstandardised second moment risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using the second moment (such as variance or semi-variance) of the return distribution, without normalising by the variance. All concrete unstandardised second moment risk measure algorithms should subtype `UnstandardisedSecondMomentAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`SecondLowerMoment`](@ref)
  - [`SecondCentralMoment`](@ref)
"""
abstract type UnstandardisedSecondMomentAlgorithm <:
              UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
    struct SecondLowerMoment{T1} <: UnstandardisedSecondMomentAlgorithm
        alg::T1
    end

Represents the second lower moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the second lower moment (semi-variance or semi-standard deviation), which quantifies downside risk by considering only deviations below a target value.

# Fields

  - `alg`: The second moment formulation algorithm used to compute the risk measure.

# Constructors

    SecondLowerMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

# Formulations

Depending on the `alg` field, this can represent either the semi-variance or semi-standard deviation.

## `SOCRiskExpr`

Computes the semi-standard deviation (square root of semi-variance) of the returns below the target.

## `QuadRiskExpr`, `SquaredSOCRiskExpr`, `RSOCRiskExpr`

Computes the semi-variance of the returns below the target.

# Examples

```jldoctest
julia> SecondLowerMoment()
SecondLowerMoment
  alg ┴ SquaredSOCRiskExpr()
```

# Related

  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct SecondLowerMoment{T1} <: UnstandardisedSecondMomentAlgorithm
    alg::T1
    function SecondLowerMoment(alg::SecondMomentFormulation)
        return new{typeof(alg)}(alg)
    end
end
function SecondLowerMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondLowerMoment(alg)
end
"""
    struct SecondCentralMoment{T1} <: UnstandardisedSecondMomentAlgorithm
        alg::T1
    end

Represents the second central moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the second central moment (variance or standard deviation), which quantifies risk by considering all deviations from a target value.

# Fields

  - `alg`: The second moment formulation algorithm used to compute the risk measure.

# Constructors

    SecondCentralMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

# Formulations

Depending on the `alg` field, this can represent either the variance or standard deviation.

## `SOCRiskExpr`

Computes the standard deviation (square root of variance) of the returns below the target.

## `QuadRiskExpr`, `SquaredSOCRiskExpr`, `RSOCRiskExpr`

Computes the variance of the returns below the target.

# Examples

```jldoctest
julia> SecondCentralMoment()
SecondCentralMoment
  alg ┴ SquaredSOCRiskExpr()
```

# Related

  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
  - [`SecondMomentFormulation`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct SecondCentralMoment{T1} <: UnstandardisedSecondMomentAlgorithm
    alg::T1
    function SecondCentralMoment(alg::SecondMomentFormulation)
        return new{typeof(alg)}(alg)
    end
end
function SecondCentralMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondCentralMoment(alg)
end
#########
struct SecondMoment{T1, T2} <: UnstandardisedSecondMomentAlgorithm
    alg1::T1
    alg2::T2
    function SecondMoment(alg1::AbstractMomentAlgorithm, alg2::SecondMomentFormulation)
        return new{typeof(alg1), typeof(alg2)}(alg1, alg2)
    end
end
function SecondMoment(; alg1::AbstractMomentAlgorithm = Full(),
                      alg2::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondMoment(alg1, alg2)
end
#########
"""
    struct StandardisedLowOrderMoment{T1, T2} <: LowOrderMomentMeasureAlgorithm
        ve::T1
        alg::T2
    end

Represents a standardised low-order moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using a low-order moment algorithm, standardised by a variance estimator. This enables risk measures such as semi-standard deviation or standardised semi-variance, which normalise the risk by the portfolio variance.

# Fields

  - `ve`: Variance estimator used for standardisation.
  - `alg`: Unstandardised second moment algorithm used to compute the risk measure.

# Constructors

    StandardisedLowOrderMoment(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                               alg::UnstandardisedSecondMomentAlgorithm = SecondLowerMoment())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> StandardisedLowOrderMoment()
StandardisedLowOrderMoment
   ve ┼ SimpleVariance
      │          me ┼ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
  alg ┼ SecondLowerMoment
      │   alg ┴ SquaredSOCRiskExpr()
```

# Related

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`UnstandardisedSecondMomentAlgorithm`](@ref)
  - [`SimpleVariance`](@ref)
  - [`SecondLowerMoment`](@ref)
"""
struct StandardisedLowOrderMoment{T1, T2} <: LowOrderMomentMeasureAlgorithm
    ve::T1
    alg::T2
    function StandardisedLowOrderMoment(ve::AbstractVarianceEstimator,
                                        alg::UnstandardisedSecondMomentAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedLowOrderMoment(;
                                    ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                   me = nothing),
                                    alg::UnstandardisedSecondMomentAlgorithm = SecondLowerMoment())
    return StandardisedLowOrderMoment(ve, alg)
end
"""
    abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end

Abstract supertype for all high-order moment-based risk measure algorithms in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using high-order statistical moments (e.g., skewness, kurtosis) of the return distribution. All concrete high-order moment risk measure algorithms should subtype `HighOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
"""
abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
    abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <: HighOrderMomentMeasureAlgorithm end

Abstract supertype for high-order moment risk measure algorithms that are not standardised by the variance in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using high-order statistical moments (such as skewness, kurtosis) without normalising by the variance. All concrete unstandardised high-order moment risk measure algorithms should subtype `UnstandardisedHighOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`ThirdLowerMoment`](@ref)
  - [`FourthLowerMoment`](@ref)
  - [`FourthCentralMoment`](@ref)
"""
abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <:
              HighOrderMomentMeasureAlgorithm end
"""
    struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end

Represents the unstandardised semi-skewness risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the third lower moment (unstandardised semi-skewness), which quantifies downside asymmetry by considering only the cubed deviations below a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
    struct FourthLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end

Represents the unstandardised semi-kurtosis risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the fourth lower moment (unstandardised semi-kurtosis), which quantifies downside tail risk by considering only the quartic deviations below a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct FourthLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
    struct FourthCentralMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end

Represents the unstandardised kurtosis risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the fourth central moment (unstandardised kurtosis), which quantifies tail risk by considering all quartic deviations from a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct FourthCentralMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
###########
struct FourthMoment{T1} <: UnstandardisedHighOrderMomentMeasureAlgorithm
    alg::T1
    function FourthMoment(alg::AbstractMomentAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function FourthMoment(; alg::AbstractMomentAlgorithm = Full())
    return FourthMoment(alg)
end
###########
"""
    struct StandardisedHighOrderMoment{T1, T2} <: HighOrderMomentMeasureAlgorithm
        ve::T1
        alg::T2
    end

Represents a standardised high-order moment risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using a high-order moment algorithm (such as semi-skewness or semi-kurtosis), standardised by a variance estimator. This enables risk measures such as standardised semi-skewness or standardised semi-kurtosis, which normalise the risk by the portfolio variance.

# Fields

  - `ve`: Variance estimator used for standardisation.
  - `alg`: Unstandardised high-order moment algorithm used to compute the risk measure.

# Constructors

    StandardisedHighOrderMoment(;
                                ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                                alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> StandardisedHighOrderMoment()
StandardisedHighOrderMoment
   ve ┼ SimpleVariance
      │          me ┼ nothing
      │           w ┼ nothing
      │   corrected ┴ Bool: true
  alg ┴ ThirdLowerMoment()
```

# Related

  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`SimpleVariance`](@ref)
  - [`ThirdLowerMoment`](@ref)
  - [`FourthLowerMoment`](@ref)
  - [`FourthCentralMoment`](@ref)
"""
struct StandardisedHighOrderMoment{T1, T2} <: HighOrderMomentMeasureAlgorithm
    ve::T1
    alg::T2
    function StandardisedHighOrderMoment(ve::AbstractVarianceEstimator,
                                         alg::UnstandardisedHighOrderMomentMeasureAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedHighOrderMoment(;
                                     ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                    me = nothing),
                                     alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return StandardisedHighOrderMoment(ve, alg)
end
for alg in (StandardisedLowOrderMoment, StandardisedHighOrderMoment)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights} = nothing)
                 return $(alg)(; ve = factory(alg.ve, w), alg = alg.alg)
             end
         end)
end
"""
    struct LowOrderMoment{T1, T2, T3, T4} <: RiskMeasure
        settings::T1
        w::T2
        mu::T3
        alg::T4
    end

Represents a low-order moment risk measure in PortfolioOptimisers.jl.

Computes portfolio risk using a low-order moment algorithm (such as first lower moment, mean absolute deviation, or second moment), optionally with custom weights and target values. This type is used for risk measures based on mean, variance, or related statistics.

# Fields

  - `settings`: Risk measure configuration.
  - `w`: Optional vector of observation weights.
  - `mu`: Optional target value, vector, or `VecScalar` for moment calculation that overrides the prior `mu` when provided. Also used to compute the moment target, via [`calc_moment_target`](@ref). If `nothing` it is computed from the returns series using the optional weights in `w`.
  - `alg`: Low-order moment risk measure algorithm.

# Constructors

    LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                   w::Union{Nothing, <:AbstractWeights} = nothing,
                   mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing,
                   alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())

Keyword arguments correspond to the fields above.

## Validation

  - If `mu` is not `nothing`:

      + `::Real`: `isfinite(mu)`.
      + `::AbstractVector`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.

# Formulations

Depending on the `alg` field, the risk measure is formulated using `JuMP` as follows:

## `FirstLowerMoment`

The first lower moment is computed as:

```math
\\begin{align}
\\mathrm{FirstLowerMoment}(\\boldsymbol{X}) &= \\mathbb{E}\\left[\\max \\circ \\left(\\mathbb{E}\\left[\\boldsymbol{X}\\right] - \\boldsymbol{X},\\, 0\\right)\\right]\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.
  - ``\\circ``: element-wise function application.

As an optimisation problem, it can be formulated as:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad \\mathbb{E}\\left[\\boldsymbol{d}\\right] \\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] - \\mathrm{X} \\boldsymbol{w}\\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations below the target.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

## `MeanAbsoluteDeviation`

The mean absolute deviation is computed as:

```math
\\begin{align}
\\mathrm{MeanAbsoluteDeviation}(\\boldsymbol{X}) &= \\mathbb{E}\\left[\\left\\lvert \\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right] \\right\\rvert\\right]
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

As an optimisation problem, it can be formulated as:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad 2 \\mathbb{E}\\left[\\boldsymbol{d}\\right]\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] - \\mathrm{X} \\boldsymbol{w}\\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations below the target.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

## `StandardisedLowOrderMoment`

Depending on the `alg` field the risk measure can either compute the second central moment or second lower moment.

!!! info

    Regardless of the formulation used, an auxiliary variable representing the square root of the central/lower moment is needed in order to constrain the risk or maximise the risk-adjusted return ratio. This is because quadratic constraints are not strictly convex, and the transformation needed to maximise the risk-adjusted return ratio requires affine variables in the numerator and denominator.

Both central and lower moments can be formulated as quadratic moments (variance or semi-variance) or their square roots (standard deviation or semi-standard deviation). Regardless of whether they are central or lower moments, they can be formulated in a variety of ways.

### `SecondCentralMoment`

Depending on the `alg` field, it can represent the variance (using different formulations in `JuMP`) or standard deviation.

The variance formulations are:

  - [`SquaredSOCRiskExpr`](@ref),
  - [`RSOCRiskExpr`](@ref),
  - [`QuadRiskExpr`](@ref).

It is computed as:

```math
\\begin{align}
\\mathrm{Variance}(\\boldsymbol{X}) = \\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right] \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

The standard deviation formulation is:

  - [`SOCRiskExpr`](@ref).

It is computed as:

```math
\\begin{align}
\\mathrm{StandardDeviation}(\\boldsymbol{X}) = \\sqrt{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right]} \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

#### `SquaredSOCRiskExpr`

Represents the variance using the square of a second order cone constrained variable.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\sigma^2\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{soc}``: second order cone.
  - ``\\odot``: element-wise (Hadamard) product.

#### `RSOCRiskExpr`

Represents the variance using a sum of squares formulation via a rotated second order cone.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot t\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(t,\\, 0.5,\\,\\boldsymbol{d}_s\\right) \\in K_{rsoc}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``t``: variance of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{rsoc}``: rotated second order cone.
  - ``\\odot``: element-wise (Hadamard) product.

#### `QuadRiskExpr`

Represents the variance using the deviations vector dotted with itself.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\boldsymbol{d}_s \\cdot \\boldsymbol{d}_s\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} 
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mu``: minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: element-wise (Hadamard) product.

#### `SOCRiskExpr`

Represents the standard deviation using a second order cone constrained variable.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad \\sqrt{f} \\cdot \\sigma\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mu``: minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: element-wise (Hadamard) product.
  - ``K_{soc}``: second order cone.

### `SecondLowerMoment`

Depending on the `alg` field, it can represent the semi-variance (using different formulations in `JuMP`) or semi-standard deviation.

The semi-variance formulations are:

  - [`SquaredSOCRiskExpr`](@ref),
  - [`RSOCRiskExpr`](@ref),
  - [`QuadRiskExpr`](@ref).

It is computed as:

```math
\\begin{align}
\\mathrm{Semi-Variance}(\\boldsymbol{X}) = \\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\,0\\right)^2\\right] \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.
  - ``\\circ``: element-wise function application.

The semi-standard deviation formulation is:

  - [`SOCRiskExpr`](@ref).

It is computed as:

```math
\\begin{align}
\\mathrm{Semi-StandardDeviation}(\\boldsymbol{X}) = \\sqrt{\\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\,0\\right)^2\\right]} \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.
  - ``\\circ``: element-wise function application.

#### `SquaredSOCRiskExpr`

Represents the semi-variance using the square of a second order cone constrained variable.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\sigma^2\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: semi-standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mu``: minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: element-wise (Hadamard) product.
  - ``K_{soc}``: second order cone.

#### `RSOCRiskExpr`

Represents the semi-variance using a sum of squares formulation via a rotated second order cone.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot t\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(t,\\, 0.5,\\,\\boldsymbol{d}_s\\right) \\in K_{rsoc}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``t``: semi-variance of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mu``: minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{rsoc}``: rotated second order cone.
  - ``\\odot``: element-wise (Hadamard) product.

#### `QuadRiskExpr`

Represents the semi-variance using the deviations vector dotted with itself.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\boldsymbol{d}_s \\cdot \\boldsymbol{d}_s\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} 
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mu``: minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: element-wise (Hadamard) product.

#### `SOCRiskExpr`

Represents the semi-standard deviation using a second order cone constrained variable.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad \\sqrt{f} \\cdot \\sigma\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} \\geq \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N×1` asset weights vector.
  - ``\\boldsymbol{d}``: `T×1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: semi-standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T×1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T×N` return matrix.
  - ``\\mu``: minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T×1` vector of observation weights.
  - ``f``: observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: element-wise (Hadamard) product.
  - ``K_{soc}``: second order cone.

# Functor

    (r::LowOrderMoment)(w::AbstractVector, X::AbstractMatrix;
                        fees::Union{Nothing, <:Fees} = nothing)

Computes the the low order moment risk measure as defined in `r` using portfolio weights `w`, return matrix `X`, and optional fees `fees`.

## Details

  - `r.alg` defines what low-order moment to compute.
  - The values of `r.mu` and `r.w` are optionally used to compute the moment target via [`calc_moment_target`](@ref), which is used in [`calc_deviations_vec`](@ref) to compute the deviation vector.

# Examples

```jldoctest
julia> LowOrderMoment()
LowOrderMoment
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
       alg ┴ FirstLowerMoment()
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`SecondLowerMoment`](@ref)
  - [`SecondCentralMoment`](@ref)
  - [`StandardisedLowOrderMoment`](@ref)
  - [`SquaredSOCRiskExpr`](@ref)
  - [`RSOCRiskExpr`](@ref)
  - [`QuadRiskExpr`](@ref)
  - [`SOCRiskExpr`](@ref)
"""
struct LowOrderMoment{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function LowOrderMoment(settings::RiskMeasureSettings,
                            w::Union{Nothing, <:AbstractWeights},
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real},
                                      <:VecScalar}, alg::LowOrderMomentMeasureAlgorithm)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing,
                        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())
    return LowOrderMoment(settings, w, mu, alg)
end
"""
    struct HighOrderMoment{T1, T2, T3, T4} <: HierarchicalRiskMeasure
        settings::T1
        w::T2
        mu::T3
        alg::T4
    end

Represents a high-order moment risk measure in PortfolioOptimisers.jl.

Computes portfolio risk using a high-order moment algorithm (such as semi-skewness, semi-kurtosis, or kurtosis), optionally with custom weights and target values. This type is used for risk measures based on third or fourth moments of the return distribution.

# Fields

  - `settings`: Risk measure configuration.
  - `w`: Optional vector of observation weights.
  - `mu`: Optional target value or vector `VecScalar` for moment calculation that overrides the prior `mu` when provided. Also used to compute the moment target, if not given it is computed from the returns series.
  - `alg`: High-order moment risk measure algorithm.

# Constructors

    HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                    w::Union{Nothing, <:AbstractWeights} = nothing,
                    mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing,
                    alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())

Keyword arguments correspond to the fields above.

## Validation

  - If `mu` is not `nothing`:

      + `::Real`: `isfinite(mu)`.
      + `::AbstractVector`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.

# Formulations

Depending on the `alg` field, the risk measure is can compute the third lower moment, fourth lower moment, or fourth central moment. Each can be standardised or unstandardised.

The unstandardised formulations are:

  - [`ThirdLowerMoment`](@ref),
  - [`FourthLowerMoment`](@ref),
  - [`FourthCentralMoment`](@ref).

The standardised formulations are:

  - [`StandardisedHighOrderMoment`](@ref), which uses a variance estimator and an unstandardised high-order moment algorithm.

## Unstandardised Central Moments

All unstandardised central moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^n\\right]
\\end{align}
```

## Standardised Central Moments

All standardised central moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\dfrac{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^n\\right]}{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right]^{n/2}}
\\end{align}
```

## Unstandardised Lower Moments

All unstandardised lower moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^n\\right]
\\end{align}
```

## Standardised Lower Moments

All standardised lower moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\dfrac{\\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^n\\right]}{\\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^2\\right]^{n/2}}
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.
  - ``\\circ``: element-wise function application.

# Functor

    (r::HighOrderMoment)(w::AbstractVector, X::AbstractMatrix;
                        fees::Union{Nothing, <:Fees} = nothing)

Computes the the high order moment risk measure as defined in `r` using portfolio weights `w`, return matrix `X`, and optional fees `fees`.

## Details

  - `r.alg` defines what low-order moment to compute.
  - The values of `r.mu` and `r.w` are optionally used to compute the moment target via [`calc_moment_target`](@ref), which is used in [`calc_deviations_vec`](@ref) to compute the deviation vector.

# Examples

```jldoctest
julia> HighOrderMoment()
HighOrderMoment
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
         w ┼ nothing
        mu ┼ nothing
       alg ┴ ThirdLowerMoment()
```

# Related

  - [`RiskMeasureSettings`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`ThirdLowerMoment`](@ref)
  - [`FourthLowerMoment`](@ref)
  - [`FourthCentralMoment`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
"""
struct HighOrderMoment{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function HighOrderMoment(settings::RiskMeasureSettings,
                             w::Union{Nothing, <:AbstractWeights},
                             mu::Union{Nothing, <:Real, <:AbstractVector{<:Real},
                                       <:VecScalar}, alg::HighOrderMomentMeasureAlgorithm)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Union{Nothing, <:AbstractWeights} = nothing,
                         mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}, <:VecScalar} = nothing,
                         alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return HighOrderMoment(settings, w, mu, alg)
end
"""
    calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                               <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                       ::Any, x::AbstractVector)

Compute the target value for moment calculations when neither a target value (`mu`) nor observation weights are provided in the risk measure.

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with both `w` and `mu` fields set to `nothing`.
  - `_`: Unused argument (typically asset weights, ignored in this method).
  - `x`: Returns vector.

# Returns

  - `target::eltype(x)`: The mean of the returns vector.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                                    <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                            ::Any, x::AbstractVector)
    return mean(x)
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any},
                                <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any}},
                       ::Any, x::AbstractVector)

Compute the target value for moment calculations when the risk measure provides an observation weights vector but no explicit target value (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `w` set to an observation weights vector and `mu` set to `nothing`.
  - `_`: Unused argument (typically asset weights, ignored in this method).
  - `x`: Returns vector.

# Returns

  - `target::eltype(x)`: The weighted mean of the returns vector, using the observation weights from `r.w`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing,
                                                       <:Any}}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:AbstractVector, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:AbstractVector, <:Any}},
                       w::AbstractVector, ::Any)

Compute the target value for moment calculations when the risk measure provides an explicit expected returns vector (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to an expected returns vector.
  - `w`: Asset weights vector.
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `target::eltype(w)`: The dot product of the asset weights and the expected returns vector.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}},
                       w::AbstractVector, ::Any)

Compute the target value for moment calculations when the risk measure provides a `VecScalar` as the expected returns (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to a `VecScalar` (an object with fields `v` for the expected returns vector and `s` for a scalar offset).
  - `w`: Asset weights vector.
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `target::promote_type(eltype(w), eltype(r.mu.v), typeof(r.mu.s))`: The sum of the dot product of the asset weights and the expected returns vector plus the scalar offset, `dot(w, r.mu.v) + r.mu.s`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
  - [`dot`](https://docs.julialang.org/en/v1/base/math/#Base.dot)
"""
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}},
                            w::AbstractVector, ::Any)
    return dot(w, r.mu.v) + r.mu.s
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Real, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:Real, <:Any}}, ::Any, ::Any)

Compute the target value for moment calculations when the risk measure provides a scalar value for the expected returns (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to a scalar value.
  - `::Any`: Unused argument (typically asset weights, ignored in this method).
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `target::Real`: The scalar value of `r.mu`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Real, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:Real, <:Any}}, ::Any,
                            ::Any)
    return r.mu
end
"""
    calc_deviations_vec(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::AbstractVector,
                    X::AbstractMatrix; fees::Union{Nothing, <:Fees} = nothing)

Compute the vector of deviations from the target value for moment-based risk measures.

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure specifying the moment calculation algorithm and target.
  - `w`: Asset weights vector.
  - `X`: Return matrix.
  - `fees`: Optional fees object to adjust net returns.

# Returns

  - `val::AbstractVector`: The vector of deviations between net portfolio returns and the computed moment target.

# Details

  - Computes net portfolio returns using the provided weights, return matrix, and optional fees.
  - Computes the target value for the moment calculation using [`calc_moment_target`](@ref).
  - Returns the element-wise difference between net returns and the target value.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
  - [`calc_net_returns`](@ref)
"""
function calc_deviations_vec(r::Union{<:LowOrderMoment, <:HighOrderMoment},
                             w::AbstractVector, X::AbstractMatrix,
                             fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment})(w::AbstractVector,
                                                                      X::AbstractMatrix,
                                                                      fees::Union{Nothing,
                                                                                  <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = abs.(calc_deviations_vec(r, w, X, fees))
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondLowerMoment{<:SOCRiskExpr}}})(w::AbstractVector,
                                                                                               X::AbstractMatrix,
                                                                                               fees::Union{Nothing,
                                                                                                           <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondLowerMoment{<:QuadSecondMomentFormulations}}})(w::AbstractVector,
                                                                                                                X::AbstractMatrix,
                                                                                                                fees::Union{Nothing,
                                                                                                                            <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondCentralMoment{<:SOCRiskExpr}}})(w::AbstractVector,
                                                                                                 X::AbstractMatrix,
                                                                                                 fees::Union{Nothing,
                                                                                                             <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondCentralMoment{<:QuadSecondMomentFormulations}}})(w::AbstractVector,
                                                                                                                  X::AbstractMatrix,
                                                                                                                  fees::Union{Nothing,
                                                                                                                              <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment})(w::AbstractVector,
                                                                       X::AbstractMatrix,
                                                                       fees::Union{Nothing,
                                                                                   <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 3
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthLowerMoment})(w::AbstractVector,
                                                                        X::AbstractMatrix,
                                                                        fees::Union{Nothing,
                                                                                    <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthCentralMoment})(w::AbstractVector,
                                                                          X::AbstractMatrix,
                                                                          fees::Union{Nothing,
                                                                                      <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:ThirdLowerMoment}})(w::AbstractVector,
                                                                                        X::AbstractMatrix,
                                                                                        fees::Union{Nothing,
                                                                                                    <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -mean(val) : -mean(val, r.w)
    return res / (sigma * sqrt(sigma))
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthLowerMoment}})(w::AbstractVector,
                                                                                         X::AbstractMatrix,
                                                                                         fees::Union{Nothing,
                                                                                                     <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthCentralMoment}})(w::AbstractVector,
                                                                                           X::AbstractMatrix,
                                                                                           fees::Union{Nothing,
                                                                                                       <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
###########
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondMoment{<:Semi,
                                                                        <:SOCRiskExpr}}})(w::AbstractVector,
                                                                                          X::AbstractMatrix,
                                                                                          fees::Union{Nothing,
                                                                                                      <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondMoment{<:Semi,
                                                                        <:QuadSecondMomentFormulations}}})(w::AbstractVector,
                                                                                                           X::AbstractMatrix,
                                                                                                           fees::Union{Nothing,
                                                                                                                       <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondMoment{<:Full,
                                                                        <:SOCRiskExpr}}})(w::AbstractVector,
                                                                                          X::AbstractMatrix,
                                                                                          fees::Union{Nothing,
                                                                                                      <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondMoment{<:Full,
                                                                        <:QuadSecondMomentFormulations}}})(w::AbstractVector,
                                                                                                           X::AbstractMatrix,
                                                                                                           fees::Union{Nothing,
                                                                                                                       <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment{<:Semi}})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment{<:Full}})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthMoment{<:Semi}}})(w::AbstractVector,
                                                                                            X::AbstractMatrix,
                                                                                            fees::Union{Nothing,
                                                                                                        <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthMoment{<:Full}}})(w::AbstractVector,
                                                                                            X::AbstractMatrix,
                                                                                            fees::Union{Nothing,
                                                                                                        <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
###########
for rt in (LowOrderMoment, HighOrderMoment)
    eval(quote
             function factory(r::$(rt), prior::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_factory(r.w, prior.w)
                 mu = nothing_scalar_array_factory(r.mu, prior.mu)
                 alg = factory(r.alg, w)
                 return $(rt)(; settings = r.settings, alg = alg, w = w, mu = mu)
             end
             function risk_measure_view(r::$(rt), i::AbstractVector, args...)
                 mu = nothing_scalar_array_view(r.mu, i)
                 return $(rt)(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
             end
         end)
end

export FirstLowerMoment, SecondLowerMoment, SecondCentralMoment, SecondMoment,
       MeanAbsoluteDeviation, ThirdLowerMoment, FourthMoment, FourthLowerMoment,
       FourthCentralMoment, StandardisedLowOrderMoment, StandardisedHighOrderMoment,
       LowOrderMoment, HighOrderMoment
