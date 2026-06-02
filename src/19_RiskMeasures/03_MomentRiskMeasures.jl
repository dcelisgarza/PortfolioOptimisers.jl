"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all moment-based risk measure algorithms in `PortfolioOptimisers.jl`.

Defines the interface for algorithms that compute portfolio risk using statistical moments (e.g., mean, variance, skewness, kurtosis) of the return distribution. All concrete moment risk measure algorithms should subtype `MomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
"""
abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all low-order moment-based risk measure algorithms in `PortfolioOptimisers.jl`.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments (e.g., mean, variance, mean absolute deviation) of the return distribution. All concrete low-order moment risk measure algorithms should subtype `LowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
"""
abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for low-order moment risk measure algorithms that are not standardised by the variance in `PortfolioOptimisers.jl`.

Defines the interface for algorithms that compute portfolio risk using low-order statistical moments without normalising by the variance. All concrete unstandardised low-order moment risk measure algorithms should subtype `UnstandardisedLowOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`FirstLowerMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
"""
abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the moment measure algorithm `alg` unchanged.

Identity pass-through used when a moment measure algorithm is provided in a context that calls [`factory`](@ref).

# Related

  - [`MomentMeasureAlgorithm`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::MomentMeasureAlgorithm, args...; kwargs...)::MomentMeasureAlgorithm
    return alg
end
"""
$(DocStringExtensions.TYPEDEF)

Represents the first lower moment risk measure algorithm in `PortfolioOptimisers.jl`.

Computes portfolio risk using the first lower moment, which is the negative mean of the deviations of the returns series below a target value.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Represents the mean absolute deviation risk measure algorithm in `PortfolioOptimisers.jl`.

Computes portfolio risk as the mean of the absolute deviations of the returns series from a target value.

# Related

  - [`UnstandardisedLowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMomentMeasureAlgorithm`](@ref)
  - [`LowOrderMoment`](@ref)
"""
struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Represents a second moment (variance or standard deviation) risk measure algorithm in `PortfolioOptimisers.jl`.

Computes portfolio risk using the second central (full) or lower (semi) moment of the return distribution, supporting both full and semi (downside) formulations. The specific formulation is determined by the `alg1` and `alg2` fields, enabling flexible representation of variance, semi-variance, standard deviation, or semi-standard deviation.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SecondMoment(;
        ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
        alg1::AbstractMomentAlgorithm = Full(),
        alg2::SecondMomentFormulation = SquaredSOCRiskExpr(),
    ) -> SecondMoment

Keywords correspond to the struct's fields.

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
@concrete struct SecondMoment <: LowOrderMomentMeasureAlgorithm
    """
    $(field_dict[:ve])
    """
    ve
    """
    $(field_dict[:alg1])
    """
    alg1
    """
    $(field_dict[:alg2])
    """
    alg2
    function SecondMoment(ve::AbstractVarianceEstimator, alg1::AbstractMomentAlgorithm,
                          alg2::SecondMomentFormulation)
        return new{typeof(ve), typeof(alg1), typeof(alg2)}(ve, alg1, alg2)
    end
end
function SecondMoment(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                      alg1::AbstractMomentAlgorithm = Full(),
                      alg2::SecondMomentFormulation = SquaredSOCRiskExpr())::SecondMoment
    return SecondMoment(ve, alg1, alg2)
end
@concrete struct EvenMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm
    """
    Order of the even moment.
    """
    p
    """
    $(field_dict[:ddof])
    """
    ddof
    """
    $(field_dict[:malg])
    """
    alg
    function EvenMoment(p::Integer, ddof::Integer, alg::AbstractMomentAlgorithm)
        @argcheck(p >= 2, DomainError)
        assert_nonempty_nonneg_finite_val(ddof, :ddof)
        return new{typeof(p), typeof(ddof), typeof(alg)}(p, ddof, alg)
    end
end
function EvenMoment(; p::Integer = 2, ddof::Integer = 0,
                    alg::AbstractMomentAlgorithm = Full())::EvenMoment
    return EvenMoment(p, ddof, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`SecondMoment`](@ref) with observation weights `w` applied to the underlying variance estimator.

# Related

  - [`SecondMoment`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::SecondMoment, w::ObsWeights)::SecondMoment
    return SecondMoment(; ve = factory(alg.ve, w), alg1 = alg.alg1, alg2 = alg.alg2)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all high-order moment-based risk measure algorithms in `PortfolioOptimisers.jl`.

Defines the interface for algorithms that compute portfolio risk using high-order statistical moments (e.g., skewness, kurtosis) of the return distribution. All concrete high-order moment risk measure algorithms should subtype `HighOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`StandardisedHighOrderMoment`](@ref)
"""
abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for high-order moment risk measure algorithms that are not standardised by the variance in `PortfolioOptimisers.jl`.

Defines the interface for algorithms that compute portfolio risk using high-order statistical moments (such as skewness, kurtosis) without normalising by the variance. All concrete unstandardised high-order moment risk measure algorithms should subtype `UnstandardisedHighOrderMomentMeasureAlgorithm` to ensure consistency and composability within the risk measure framework.

# Related Types

  - [`ThirdLowerMoment`](@ref)
  - [`FourthMoment`](@ref)
"""
abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <:
              HighOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Represents the unstandardised semi-skewness risk measure algorithm in `PortfolioOptimisers.jl`.

Computes portfolio risk using the third lower moment (unstandardised semi-skewness), which quantifies downside asymmetry by considering only the cubed deviations below a target value. This algorithm is unstandardised and operates directly on the return distribution.

# Related

  - [`UnstandardisedHighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMomentMeasureAlgorithm`](@ref)
  - [`HighOrderMoment`](@ref)
"""
struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Represents the unstandardised fourth moment (kurtosis or semi-kurtosis) risk measure algorithm in `PortfolioOptimisers.jl`.

Computes portfolio risk using the fourth central (full) or lower (semi) moment of the return distribution, depending on the provided moment algorithm. This algorithm quantifies the "tailedness" of the return distribution without normalising by the variance.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FourthMoment(;
        alg::AbstractMomentAlgorithm = Full(),
    ) -> FourthMoment

Keywords correspond to the struct's fields.

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
@concrete struct FourthMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm
    """
    $(field_dict[:malg])
    """
    alg
    function FourthMoment(alg::AbstractMomentAlgorithm)
        return new{typeof(alg)}(alg)
    end
end
function FourthMoment(; alg::AbstractMomentAlgorithm = Full())::FourthMoment
    return FourthMoment(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a standardised high-order moment risk measure algorithm in `PortfolioOptimisers.jl`.

Computes portfolio risk using a high-order moment algorithm (such as semi-skewness or semi-kurtosis), standardised by a variance estimator.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    StandardisedHighOrderMoment(;
        ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
        alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
    ) -> StandardisedHighOrderMoment

Keywords correspond to the struct's fields.

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
@concrete struct StandardisedHighOrderMoment <: HighOrderMomentMeasureAlgorithm
    """
    $(field_dict[:ve])
    """
    ve
    """
    $(field_dict[:malg])
    """
    alg
    function StandardisedHighOrderMoment(ve::AbstractVarianceEstimator,
                                         alg::UnstandardisedHighOrderMomentMeasureAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedHighOrderMoment(;
                                     ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                    me = nothing),
                                     alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())::StandardisedHighOrderMoment
    return StandardisedHighOrderMoment(ve, alg)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new [`StandardisedHighOrderMoment`](@ref) with observation weights `w` applied to the underlying variance estimator.

# Related

  - [`StandardisedHighOrderMoment`](@ref)
  - [`factory`](@ref)
"""
function factory(alg::StandardisedHighOrderMoment,
                 w::ObsWeights)::StandardisedHighOrderMoment
    return StandardisedHighOrderMoment(; ve = factory(alg.ve, w), alg = alg.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a low-order moment risk measure in `PortfolioOptimisers.jl`.

Computes portfolio risk using a low-order moment algorithm (such as first lower moment, mean absolute deviation, or second moment), optionally with custom weights and target values. This type is used for risk measures based on mean, variance, or related statistics.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LowOrderMoment(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:Num_VecNum_VecScalar} = nothing,
        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment(),
    ) -> LowOrderMoment

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is not `nothing`:

      + `::Number`: `isfinite(mu)`.
      + `::AbstractVector`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.

# Mathematical definition

Depending on the `alg` field, the risk measure is formulated using `JuMP` as follows:

## `FirstLowerMoment`

The first lower moment is computed as:

```math
\\begin{align}
\\mathrm{FirstLowerMoment}(\\boldsymbol{X}) &= \\mathbb{E}\\left[\\max \\circ \\left(\\mathbb{E}\\left[\\boldsymbol{X}\\right] - \\boldsymbol{X},\\, 0\\right)\\right]\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.
  - ``\\circ``: Element-wise function application.

As an optimisation problem, it can be formulated as:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & \\mathbb{E}\\left[\\boldsymbol{d}\\right] \\\\
\\mathrm{s.t.} \\quad & \\boldsymbol{d} \\geq \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] - \\mathrm{X} \\boldsymbol{w}\\\\
               \\quad & \\boldsymbol{d} \\geq 0 \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations below the target.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.

## `MeanAbsoluteDeviation`

The mean absolute deviation is computed as:

```math
\\begin{align}
\\mathrm{MeanAbsoluteDeviation}(\\boldsymbol{X}) &= \\mathbb{E}\\left[\\left\\lvert \\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right] \\right\\rvert\\right]\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.

As an optimisation problem, it can be formulated as:

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & 2 \\mathbb{E}\\left[\\boldsymbol{d}\\right]\\\\
\\mathrm{s.t.} \\quad & \\boldsymbol{d} \\geq \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] - \\mathrm{X} \\boldsymbol{w}\\\\
               \\quad & \\boldsymbol{d} \\geq 0 \\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations below the target.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.

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

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.

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

  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.

#### `SquaredSOCRiskExpr`

Represents the (semi-)variance using the square of a second order cone constrained variable.

The variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & f \\cdot \\sigma^2\\\\
\\mathrm{s.t.} \\quad & \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               \\quad & \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: Standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{soc}``: Second order cone.
  - ``\\odot``: Element-wise (Hadamard) product.

The semi-variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & f \\cdot \\sigma^2\\\\
\\mathrm{s.t.} \\quad & \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               \\quad & \\boldsymbol{d} \\geq 0 \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               \\quad & \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: Standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{soc}``: Second order cone.
  - ``\\odot``: Element-wise (Hadamard) product.

#### `RSOCRiskExpr`

Represents the (semi-)variance using a sum of squares formulation via a rotated second order cone.

The variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & f \\cdot t\\\\
\\mathrm{s.t.} \\quad & \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               \\quad & \\left(t,\\, 0.5,\\,\\boldsymbol{d}_s\\right) \\in K_{rsoc}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``t``: Variance of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{rsoc}``: Rotated second order cone.
  - ``\\odot``: Element-wise (Hadamard) product.

The semi-variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & f \\cdot t\\\\
\\mathrm{s.t.} \\quad & \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               \\quad & \\boldsymbol{d} \\geq 0 \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               \\quad & \\left(t,\\, 0.5,\\,\\boldsymbol{d}_s\\right) \\in K_{rsoc}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``t``: Variance of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{rsoc}``: Rotated second order cone.
  - ``\\odot``: Element-wise (Hadamard) product.

#### `QuadRiskExpr`

Represents the (semi-)variance using the deviations vector dotted with itself.

The variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & f \\cdot \\boldsymbol{d}_s \\cdot \\boldsymbol{d}_s\\\\
\\mathrm{s.t.} \\quad & \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: Element-wise (Hadamard) product.

The semi-variance is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & f \\cdot \\boldsymbol{d}_s \\cdot \\boldsymbol{d}_s\\\\
\\mathrm{s.t.} \\quad & \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               \\quad & \\boldsymbol{d} \\geq 0 \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\mu``: Minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: Element-wise (Hadamard) product.

#### `SOCRiskExpr`

Represents the (semi-)standard deviation using a second order cone constrained variable.

The standard deviation is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & \\sqrt{f} \\cdot \\sigma\\\\
\\mathrm{s.t.} \\quad & \\boldsymbol{d} = \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               \\quad & \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: Standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``K_{soc}``: Second order cone.
  - ``\\odot``: Element-wise (Hadamard) product.

The semi-standard deviation is formulated as.

```math
\\begin{align}
\\underset{\\boldsymbol{w},\\,\\boldsymbol{d}}{\\mathrm{opt}} \\quad & \\sqrt{f} \\cdot \\sigma\\\\
\\mathrm{s.t.} \\quad & \\mathrm{X} \\boldsymbol{w} - \\mathbb{E}\\left[\\mathrm{X} \\boldsymbol{w}\\right] \\geq -\\boldsymbol{d} \\\\
               \\quad & \\boldsymbol{d} \\geq 0 \\\\
               \\quad & \\boldsymbol{d}_s = \\sqrt{\\boldsymbol{\\lambda}} \\odot \\boldsymbol{d} \\\\
               \\quad & \\left(\\sigma,\\, \\boldsymbol{d}_s\\right) \\in K_{soc}\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}``: `N × 1` asset weights vector.
  - ``\\boldsymbol{d}``: `T × 1` vector of auxiliary decision variables representing deviations from the target.
  - ``\\sigma``: Standard deviation of the portfolio returns.
  - ``\\boldsymbol{d}_s``: `T × 1` vector of scaled deviations according to observation weights.
  - ``\\mathrm{X}``: `T × N` return matrix.
  - ``\\mu``: Minimum acceptable return.
  - ``\\boldsymbol{\\lambda}``: `T × 1` vector of observation weights.
  - ``f``: Observation weights scaling factor, it is a function of the type of observation weights.
  - ``\\odot``: Element-wise (Hadamard) product.
  - ``K_{soc}``: Second order cone.

# Functor

    (r::LowOrderMoment)(w::VecNum, X::MatNum;
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
@concrete struct LowOrderMoment <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_rm])
    """
    w
    """
    $(field_dict[:mu_rm])
    """
    mu
    """
    $(field_dict[:malg])
    """
    alg
    function LowOrderMoment(settings::RiskMeasureSettings, w::Option{<:ObsWeights},
                            mu::Option{<:Num_VecNum_VecScalar},
                            alg::LowOrderMomentMeasureAlgorithm)
        if isa(mu, VecNum)
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
                        w::Option{<:ObsWeights} = nothing,
                        mu::Option{<:Num_VecNum_VecScalar} = nothing,
                        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())::LowOrderMoment
    return LowOrderMoment(settings, w, mu, alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a high-order moment risk measure in `PortfolioOptimisers.jl`.

Computes portfolio risk using a high-order moment algorithm (such as semi-skewness, semi-kurtosis, or kurtosis), optionally with custom weights and target values. This type is used for risk measures based on third or fourth moments of the return distribution.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HighOrderMoment(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        w::Option{<:ObsWeights} = nothing,
        mu::Option{<:Num_VecNum_VecScalar} = nothing,
        alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment(),
    ) -> HighOrderMoment

Keywords correspond to the struct's fields.

## Validation

  - If `mu` is not `nothing`:

      + `::Number`: `isfinite(mu)`.
      + `::AbstractVector`: `!isempty(mu)` and `all(isfinite, mu)`.

  - If `w` is not `nothing`, `!isempty(w)`.

# Mathematical definition

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
\\mu_n &= \\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^n\\right]\\,.
\\end{align}
```

Where:

  - ``\\mu_n``: ``n``-th central moment.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.
  - ``n``: Moment order.

All unstandardised lower moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^n\\right]\\,.
\\end{align}
```

Where:

  - ``\\mu_n``: ``n``-th lower moment.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.
  - ``\\circ``: Element-wise function application.
  - ``n``: Moment order.

## Standardised Moments

All standardised central moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\dfrac{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^n\\right]}{\\mathbb{E}\\left[\\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right]\\right)^2\\right]^{n/2}}\\,.
\\end{align}
```

Where:

  - ``\\mu_n``: ``n``-th standardised central moment.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.
  - ``n``: Moment order.

All standardised lower moments have the following formula.

```math
\\begin{align}
\\mu_n &= \\dfrac{\\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^n\\right]}{\\mathbb{E}\\left[\\min \\circ \\left(\\boldsymbol{X} - \\mathbb{E}\\left[\\boldsymbol{X}\\right],\\, 0\\right)^2\\right]^{n/2}}\\,.
\\end{align}
```

Where:

  - ``\\mu_n``: ``n``-th standardised lower moment.
  - ``\\boldsymbol{X}``: `T × 1` vector of portfolio returns.
  - ``\\mathbb{E}[\\cdot]``: Expected value operator, supports weighted averages.
  - ``\\circ``: Element-wise function application.
  - ``n``: Moment order.

# Functor

    (r::HighOrderMoment)(w::VecNum, X::MatNum;
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
@concrete struct HighOrderMoment <: HierarchicalRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:w_rm])
    """
    w
    """
    $(field_dict[:mu_rm])
    """
    mu
    """
    $(field_dict[:malg])
    """
    alg
    function HighOrderMoment(settings::RiskMeasureSettings, w::Option{<:ObsWeights},
                             mu::Option{<:Num_VecNum_VecScalar},
                             alg::HighOrderMomentMeasureAlgorithm)
        if isa(mu, VecNum)
            @argcheck(!isempty(mu))
            @argcheck(all(isfinite, mu))
        elseif isa(mu, Number)
            @argcheck(isfinite(mu))
        end
        assert_nonempty_nonneg_finite_val(w, :w)
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:ObsWeights} = nothing,
                         mu::Option{<:Num_VecNum_VecScalar} = nothing,
                         alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())::HighOrderMoment
    return HighOrderMoment(settings, w, mu, alg)
end
"""
    const LoHiOrderMoment{T1, T2, T3, T4} = Union{...}

Parameterised union of [`LowOrderMoment`](@ref) and [`HighOrderMoment`](@ref) sharing the same type parameters.

Used for unified dispatch on moment-target calculation methods.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
const LoHiOrderMoment{T1, T2, T3, T4} = Union{<:LowOrderMoment{T1, T2, T3, T4},
                                              <:HighOrderMoment{T1, T2, T3, T4}}
"""
    calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any},
                       ::Any, x::VecNum)

Compute the target value for moment calculations when neither a target value (`mu`) nor observation weights are provided in the risk measure.

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with both `w` and `mu` fields set to `nothing`.
  - `_`: Unused argument (typically asset weights, ignored in this method).
  - `x`: Returns vector.

# Returns

  - `tgt::eltype(x)`: The mean of the returns vector.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any}, ::Any,
                            x::VecNum)
    return Statistics.mean(x)
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing, <:Any},
                       ::Any, x::VecNum)

Compute the target value for moment calculations when the risk measure provides an observation weights vector but no explicit target value (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `w` set to an observation weights vector and `mu` set to `nothing`.
  - `_`: Unused argument (typically asset weights, ignored in this method).
  - `x`: Returns vector.

# Returns

  - `tgt::eltype(x)`: The weighted mean of the returns vector, using the observation weights from `r.w`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing,
                                               <:Any}, ::Any, x::VecNum)
    return Statistics.mean(x, r.w)
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any},
                       w::VecNum, ::Any)

Compute the target value for moment calculations when the risk measure provides an explicit expected returns vector (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to an expected returns vector.
  - `w`: Asset weights vector.
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `tgt::eltype(w)`: The dot product of the asset weights and the expected returns vector.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any}, w::VecNum,
                            ::Any)
    return LinearAlgebra.dot(w, r.mu)
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                       w::VecNum, ::Any)

Compute the target value for moment calculations when the risk measure provides a `VecScalar` as the expected returns (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to a `VecScalar` (an object with fields `v` for the expected returns vector and `s` for a scalar offset).
  - `w`: Asset weights vector.
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `tgt::promote_type(eltype(w), eltype(r.mu.v), typeof(r.mu.s))`: The sum of the dot product of the asset weights and the expected returns vector plus the scalar offset, `LinearAlgebra.dot(w, r.mu.v) + r.mu.s`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
  - [`dot`](https://docs.julialang.org/en/v1/base/math/#Base.dot)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}, w::VecNum,
                            ::Any)
    return LinearAlgebra.dot(w, r.mu.v) + r.mu.s
end
"""
    calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any},
                       ::Any, ::Any)

Compute the target value for moment calculations when the risk measure provides a scalar value for the expected returns (`mu`).

# Arguments

  - `r`: A `LowOrderMoment` or `HighOrderMoment` risk measure with `mu` set to a scalar value.
  - `::Any`: Unused argument (typically asset weights, ignored in this method).
  - `::Any`: Unused argument (typically the returns vector, ignored in this method).

# Returns

  - `tgt::Number`: The scalar value of `r.mu`.

# Related

  - [`LowOrderMoment`](@ref)
  - [`HighOrderMoment`](@ref)
  - [`calc_moment_target`](@ref)
"""
function calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
    return r.mu
end
"""
    calc_deviations_vec(r::LoHiOrderMoment, w::VecNum,
                    X::MatNum; fees::Option{<:Fees} = nothing)

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
function calc_deviations_vec(r::LoHiOrderMoment, w::VecNum, X::MatNum,
                             fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    tgt = calc_moment_target(r, w, x)
    return x .- tgt
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:FirstLowerMoment})(w::VecNum, X::MatNum,
                                                 fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return isnothing(r.w) ? -Statistics.mean(val) : -Statistics.mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:MeanAbsoluteDeviation})(w::VecNum, X::MatNum,
                                                      fees::Option{<:Fees} = nothing)
    val = abs.(calc_deviations_vec(r, w, X, fees))
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                             <:ThirdLowerMoment})(w::VecNum, X::MatNum,
                                                  fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 3
    return isnothing(r.w) ? -Statistics.mean(val) : -Statistics.mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:ThirdLowerMoment}})(w::VecNum,
                                                                                        X::MatNum,
                                                                                        fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -Statistics.mean(val) : -Statistics.mean(val, r.w)
    return res / (sigma * sqrt(sigma))
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:SecondMoment{<:Any, <:Semi, <:SOCRiskExpr}})(w::VecNum,
                                                                           X::MatNum,
                                                                           fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:SecondMoment{<:Any, <:Semi, <:QuadSecondMomentFormulations}})(w::VecNum,
                                                                                            X::MatNum,
                                                                                            fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:SecondMoment{<:Any, <:Full, <:SOCRiskExpr}})(w::VecNum,
                                                                           X::MatNum,
                                                                           fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:SecondMoment{<:Any, <:Full, <:QuadSecondMomentFormulations}})(w::VecNum,
                                                                                            X::MatNum,
                                                                                            fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:EvenMoment{<:Any, <:Any, <:Semi}})(w::VecNum, X::MatNum,
                                                                 fees::Option{<:Fees} = nothing)
    T = size(X, 1) - r.alg.ddof
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val = if isnothing(r.w)
        LinearAlgebra.norm(val, 2 * r.alg.p)
    else
        T = T / size(X, 1) * sum(r.w)
        LinearAlgebra.norm(val .* r.w, 2 * r.alg.p)
    end
    return val^2 / T^inv(r.alg.p)
end
function (r::LowOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                            <:EvenMoment{<:Any, <:Any, <:Full}})(w::VecNum, X::MatNum,
                                                                 fees::Option{<:Fees} = nothing)
    T = size(X, 1) - r.alg.ddof
    val = calc_deviations_vec(r, w, X, fees)
    val = if isnothing(r.w)
        LinearAlgebra.norm(val, 2 * r.alg.p)
    else
        T = T / size(X, 1) * sum(r.w)
        LinearAlgebra.norm(val .* r.w, 2 * r.alg.p)
    end
    return val^2 / T^inv(r.alg.p)
end
function (r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                             <:FourthMoment{<:Semi}})(w::VecNum, X::MatNum,
                                                      fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                             <:FourthMoment{<:Full}})(w::VecNum, X::MatNum,
                                                      fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthMoment{<:Semi}}})(w::VecNum,
                                                                                            X::MatNum,
                                                                                            fees::Option{<:Fees} = nothing)
    val = min.(calc_deviations_vec(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:Option{<:StatsBase.AbstractWeights}, <:Any,
                             <:StandardisedHighOrderMoment{<:Any, <:FourthMoment{<:Full}}})(w::VecNum,
                                                                                            X::MatNum,
                                                                                            fees::Option{<:Fees} = nothing)
    val = calc_deviations_vec(r, w, X, fees)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? Statistics.mean(val) : Statistics.mean(val, r.w)
    return res / sigma^2
end
function (r::LowOrderMoment{<:Any, <:DynamicAbstractWeights, <:Any, <:Any})(w::VecNum,
                                                                            X::MatNum,
                                                                            fees::Option{<:Fees} = nothing)
    return LowOrderMoment(; settings = r.settings, alg = r.alg,
                          w = get_observation_weights(r.w, X), mu = r.mu)(w, X, fees)
end
function (r::HighOrderMoment{<:Any, <:DynamicAbstractWeights, <:Any, <:Any})(w::VecNum,
                                                                             X::MatNum,
                                                                             fees::Option{<:Fees} = nothing)
    return HighOrderMoment(; settings = r.settings, alg = r.alg,
                           w = get_observation_weights(r.w, X), mu = r.mu)(w, X, fees)
end
for rt in (LowOrderMoment, HighOrderMoment)
    eval(quote
             function factory(r::$(rt), pr::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_selector(r.w, pr.w)
                 mu = nothing_scalar_array_selector(r.mu, pr.mu)
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
       FourthMoment, StandardisedHighOrderMoment, LowOrderMoment, HighOrderMoment,
       EvenMoment
