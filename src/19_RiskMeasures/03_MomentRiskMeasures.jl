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
"""
abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
    abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end

Abstract supertype for low-order moment risk measure algorithms that are not standardised by the variance in PortfolioOptimisers.jl.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments without normalising by the variance. All concrete unstandardised low-order moment risk measure algorithms should subtype `UnstandardisedLowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
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
    struct SecondMoment{T1, T2, T3} <: LowOrderMomentMeasureAlgorithm
        ve::T1
        alg1::T2
        alg2::T3
    end

Represents a second moment (variance or standard deviation) risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the second central (full) or lower (semi) moment of the return distribution, supporting both full and semi (downside) formulations. The specific formulation is determined by the `alg1` and `alg2` fields, enabling flexible representation of variance, semi-variance, standard deviation, or semi-standard deviation.

# Fields

  - `ve`: Variance estimator used to compute the second moment.
  - `alg1`: Moment algorithm specifying whether to use all deviations or only downside deviations.
  - `alg2`: Second moment formulation specifying the optimisation formulation.

# Constructors

    SecondMoment(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                 alg1::AbstractMomentAlgorithm = Full(),
                 alg2::SecondMomentFormulation = SquaredSOCRiskExpr())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> SecondMoment()
SecondMoment
    ve ┼ SimpleVariance
       │          me ┼ nothing
       │           w ┼ nothing
       │   corrected ┴ Bool: true
  alg1 ┼ Full()
  alg2 ┴ SquaredSOCRiskExpr()
```

# Related

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`AbstractVarianceEstimator`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
  - [`SecondMomentFormulation`](@ref)
"""
struct SecondMoment{T1, T2, T3} <: LowOrderMomentMeasureAlgorithm
    ve::T1
    alg1::T2
    alg2::T3
    function SecondMoment(ve::AbstractVarianceEstimator, alg1::AbstractMomentAlgorithm,
                          alg2::SecondMomentFormulation)
        return new{typeof(ve), typeof(alg1), typeof(alg2)}(ve, alg1, alg2)
    end
end
function SecondMoment(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                      alg1::AbstractMomentAlgorithm = Full(),
                      alg2::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondMoment(ve, alg1, alg2)
end
function factory(alg::SecondMoment, w::Option{<:AbstractWeights} = nothing)
    return SecondMoment(; ve = factory(alg.ve, w), alg1 = alg.alg1, alg2 = alg.alg2)
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
  - [`FourthMoment`](@ref)
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
    struct FourthMoment{T1} <: UnstandardisedHighOrderMomentMeasureAlgorithm
        alg::T1
    end

Represents the unstandardised fourth moment (kurtosis or semi-kurtosis) risk measure algorithm in PortfolioOptimisers.jl.

Computes portfolio risk using the fourth central (full) or lower (semi) moment of the return distribution, depending on the provided moment algorithm. This algorithm quantifies the "tailedness" of the return distribution without normalising by the variance.

# Fields

  - `alg`: Moment algorithm specifying whether to use all deviations or only downside deviations.

# Constructors

    FourthMoment(; alg::AbstractMomentAlgorithm = Full())

Keyword arguments correspond to the fields above.

# Examples

```jldoctest
julia> FourthMoment()
FourthMoment
  alg ┴ Full()
```

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`AbstractMomentAlgorithm`](@ref)
"""
struct FourthMoment{T1} <: UnstandardisedHighOrderMomentMeasureAlgorithm
    alg::T1
    function FourthMoment(alg::AbstractMomentAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function FourthMoment(; alg::AbstractMomentAlgorithm = Full())
    return FourthMoment(alg)
end
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
  - [`AbstractVarianceEstimator`](@ref)
  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
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
function factory(alg::StandardisedHighOrderMoment, w::Option{<:AbstractWeights} = nothing)
    return StandardisedHighOrderMoment(; ve = factory(alg.ve, w), alg = alg.alg)
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
  - `mu`: Optional target scalar, vector, or `VecScalar` value for moment calculation that overrides the prior `mu` when provided. Also used to compute the moment target, via [`calc_moment_target`](@ref). If `nothing` it is computed from the returns series using the optional weights in `w`.
  - `alg`: Low-order moment risk measure algorithm.

# Constructors

    LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                   w::Option{<:AbstractWeights} = nothing,
                   mu::Option{<:Union{<:NumUNumVec, <:VecScalar}} = nothing,
                   alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())

Keyword arguments correspond to the fields above.

## Validation

  - If `mu` is not `nothing`:

      + `::Number`: `isfinite(mu)`.
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

## `SecondMoment`

Depending on the `alg1` field the risk measure can either compute the second central moment or second lower moment.

!!! info

    Regardless of the formulation used, an auxiliary variable representing the square root of the central/lower moment is needed in order to constrain the risk or maximise the risk-adjusted return ratio. This is because quadratic constraints are not strictly convex, and the transformation needed to maximise the risk-adjusted return ratio requires affine variables in the numerator and denominator.

Both central and lower moments can be formulated as quadratic moments (variance or semi-variance) or their square roots (standard deviation or semi-standard deviation). Regardless of whether they are central or lower moments, they can be formulated in a variety of ways.

Depending on the `alg2` field, it can represent the variance (using different formulations in `JuMP`) or standard deviation.

The (semi-)variance formulations are:

  - [`SquaredSOCRiskExpr`](@ref),
  - [`RSOCRiskExpr`](@ref),
  - [`QuadRiskExpr`](@ref).

It is computed as:

```math
\\begin{align}
\\mathrm{Variance}(\\boldsymbol{X}) &= \\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right] \\,.
\\mathrm{Semi-Variance}(\\boldsymbol{X}) &= \\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\,0\\right)^2\\right] \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

The (semi-)standard deviation formulation is:

  - [`SOCRiskExpr`](@ref).

It is computed as:

```math
\\begin{align}
\\mathrm{StandardDeviation}(\\boldsymbol{X}) &= \\sqrt{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right]} \\,.
\\mathrm{Semi-StandardDeviation}(\\boldsymbol{X}) &= \\sqrt{\\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\,0\\right)^2\\right]} \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T×1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: expected value operator, supports weighted averages.

#### `SquaredSOCRiskExpr`

Represents the (semi-)variance using the square of a second order cone constrained variable.

The variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\sigma^2\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}
\\end{align}
```

The semi-variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\sigma^2\\\\
\\mathrm{s.t.} &\\qquad \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
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

Represents the (semi-)variance using a sum of squares formulation via a rotated second order cone.

The variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot t\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(t,\\, 0.5,\\,\\boldsymbol{d}_s\\right) \\in K_{rsoc}
\\end{align}
```

The semi-variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot t\\\\
\\mathrm{s.t.} &\\qquad \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
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

Represents the (semi-)variance using the deviations vector dotted with itself.

The variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\boldsymbol{d}_s \\cdot \\boldsymbol{d}_s\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} 
\\end{align}
```

The semi-variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad f \\cdot \\boldsymbol{d}_s \\cdot \\boldsymbol{d}_s\\\\
\\mathrm{s.t.} &\\qquad \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
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

Represents the (semi-)standard deviation using a second order cone constrained variable.

The standard deviation is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad \\sqrt{f} \\cdot \\sigma\\\\
\\mathrm{s.t.} &\\qquad \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               &\\qquad \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               &\\qquad \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}
\\end{align}
```

The semi-standard deviation is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} &\\qquad \\sqrt{f} \\cdot \\sigma\\\\
\\mathrm{s.t.} &\\qquad \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               &\\qquad \\boldsymbol{d} \\geq 0 \\\\
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

# Functor

    (r::LowOrderMoment)(w::NumVec, X::NumMat;
                        fees::Option{<:Fees} = nothing)

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
  - [`SecondMoment`](@ref)
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
    function LowOrderMoment(settings::RiskMeasureSettings, w::Option{<:AbstractWeights},
                            mu::Union{Nothing, <:NumUNumVec, <:VecScalar},
                            alg::LowOrderMomentMeasureAlgorithm)
        if isa(mu, NumVec)
            @argcheck(!isempty(mu))
            @argcheck(all(isfinite, mu))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu))
        end
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        w::Option{<:AbstractWeights} = nothing,
                        mu::Option{<:Union{<:NumUNumVec, <:VecScalar}} = nothing,
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
  - `mu`: Optional target scalar, vector, or `VecScalar` value for moment calculation that overrides the prior `mu` when provided. Also used to compute the moment target, via [`calc_moment_target`](@ref). If `nothing` it is computed from the returns series using the optional weights in `w`.
  - `alg`: High-order moment risk measure algorithm.

# Constructors

    HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                    w::Option{<:AbstractWeights} = nothing,
                    mu::Option{<:Union{<:NumUNumVec, <:VecScalar}} = nothing,
                    alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())

Keyword arguments correspond to the fields above.

## Validation

  - If `mu` is not `nothing`:

      + `::Number`: `isfinite(mu)`.
      + `::AbstractVector`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.

# Formulations

Depending on the `alg` field, the risk measure is can compute the third lower moment, fourth lower (semi) moment, or fourth central (full) moment. Each can be standardised or unstandardised.

The unstandardised formulations are:

  - [`ThirdLowerMoment`](@ref),
  - [`FourthMoment`](@ref),

The standardised formulations are:

  - [`StandardisedHighOrderMoment`](@ref), which uses a variance estimator and an unstandardised high-order moment algorithm.

## Unstandardised Moments

All unstandardised central moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^n\\right]
\\end{align}
```

All unstandardised lower moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^n\\right]
\\end{align}
```

## Standardised Moments

All standardised central moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\dfrac{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^n\\right]}{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right]^{n/2}}
\\end{align}
```

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

    (r::HighOrderMoment)(w::NumVec, X::NumMat;
                        fees::Option{<:Fees} = nothing)

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
  - [`FourthMoment`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
"""
struct HighOrderMoment{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function HighOrderMoment(settings::RiskMeasureSettings, w::Option{<:AbstractWeights},
                             mu::Union{Nothing, <:NumUNumVec, <:VecScalar},
                             alg::HighOrderMomentMeasureAlgorithm)
        if isa(mu, NumVec)
            @argcheck(!isempty(mu))
            @argcheck(all(isfinite, mu))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu))
        end
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:AbstractWeights} = nothing,
                         mu::Option{<:Union{<:NumUNumVec, <:VecScalar}} = nothing,
                         alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return HighOrderMoment(settings, w, mu, alg)
end
"""
    calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                               <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                       ::Any, x::NumVec)

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
                            ::Any, x::NumVec)
    return mean(x)
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any},
                                <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any}},
                       ::Any, x::NumVec)

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
                                                       <:Any}}, ::Any, x::NumVec)
    return mean(x, r.w)
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:NumVec, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:NumVec, <:Any}},
                       w::NumVec, ::Any)

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
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:NumVec, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:NumVec, <:Any}},
                            w::NumVec, ::Any)
    return dot(w, r.mu)
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}},
                       w::NumVec, ::Any)

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
                            w::NumVec, ::Any)
    return dot(w, r.mu.v) + r.mu.s
end
"""
    calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Number, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:Number, <:Any}}, ::Any, ::Any)

Compute the target value for moment calculations when the risk measure provides a scalar value for the expected returns (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to a scalar value.
  - `::Any`: Unused argument (typically asset weights, ignored in this method).
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `target::Number`: The scalar value of `r.mu`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Number, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:Number, <:Any}},
                            ::Any, ::Any)
    return r.mu
end
"""
    calc_deviations_vec(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::NumVec,
                    X::NumMat; fees::Option{<:Fees} = nothing)

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
function calc_deviations_vec(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::NumVec,
                             X::NumMat, fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment})(w::NumVec, X::NumMat,
                                                                      fees::Union{Nothing,
                                                                                  <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation})(w::NumVec,
                                                                           X::NumMat,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = abs.(calc_deviations_vec(r, w, X, fees))
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment})(w::NumVec, X::NumMat,
                                                                       fees::Union{Nothing,
                                                                                   <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 3
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:ThirdLowerMoment}})(w::NumVec,
                                                                                        X::NumMat,
                                                                                        fees::Union{Nothing,
                                                                                                    <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -mean(val) : -mean(val, r.w)
    return res / (sigma * sqrt(sigma))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:SecondMoment{<:Any, <:Semi, <:SOCRiskExpr}})(w::NumVec,
                                                                           X::NumMat,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:SecondMoment{<:Any, <:Semi, <:QuadSecondMomentFormulations}})(w::NumVec,
                                                                                            X::NumMat,
                                                                                            fees::Union{Nothing,
                                                                                                        <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:SecondMoment{<:Any, <:Full, <:SOCRiskExpr}})(w::NumVec,
                                                                           X::NumMat,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:SecondMoment{<:Any, <:Full, <:QuadSecondMomentFormulations}})(w::NumVec,
                                                                                            X::NumMat,
                                                                                            fees::Union{Nothing,
                                                                                                        <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment{<:Semi}})(w::NumVec,
                                                                           X::NumMat,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthMoment{<:Full}})(w::NumVec,
                                                                           X::NumMat,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthMoment{<:Semi}}})(w::NumVec,
                                                                                            X::NumMat,
                                                                                            fees::Union{Nothing,
                                                                                                        <:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthMoment{<:Full}}})(w::NumVec,
                                                                                            X::NumMat,
                                                                                            fees::Union{Nothing,
                                                                                                        <:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
for rt in (LowOrderMoment, HighOrderMoment)
    eval(quote
             function factory(r::$(rt), prior::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_factory(r.w, prior.w)
                 mu = nothing_scalar_array_factory(r.mu, prior.mu)
                 alg = factory(r.alg, w)
                 return $(rt)(; settings = r.settings, alg = alg, w = w, mu = mu)
             end
             function risk_measure_view(r::$(rt), i, args...)
                 mu = nothing_scalar_array_view(r.mu, i)
                 return $(rt)(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
             end
         end)
end

export FirstLowerMoment, SecondMoment, MeanAbsoluteDeviation, ThirdLowerMoment,
       FourthMoment, StandardisedHighOrderMoment, LowOrderMoment, HighOrderMoment
