"""
$(DocStringExtensions.TYPEDEF)

Represents a risk ratio risk measure for hierarchical portfolio optimisation.

`RiskRatioRiskMeasure` computes the ratio of two risk measures, enabling the construction of risk-adjusted performance metrics for use in hierarchical optimisation routines.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RiskRatio}(\\boldsymbol{x}) &= \\frac{r_1(\\boldsymbol{x})}{r_2(\\boldsymbol{x})}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RiskRatio}(\\boldsymbol{x})``: Risk ratio of the portfolio.
  - $(math_dict[:xret])
  - ``r_1``: First (numerator) optimisation risk measure.
  - ``r_2``: Second (denominator) optimisation risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskRatioRiskMeasure(;
        r1::OptimisationRiskMeasure = Variance(),
        r2::OptimisationRiskMeasure = ConditionalValueatRisk()
    ) -> RiskRatioRiskMeasure

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> RiskRatioRiskMeasure()
RiskRatioRiskMeasure
  r1 ┼ Variance
     │   settings ┼ RiskMeasureSettings
     │            │   scale ┼ Float64: 1.0
     │            │      ub ┼ nothing
     │            │     rke ┴ Bool: true
     │      sigma ┼ nothing
     │       chol ┼ nothing
     │         rc ┼ nothing
     │        alg ┴ SquaredSOCRiskExpr()
  r2 ┼ ConditionalValueatRisk
     │   settings ┼ RiskMeasureSettings
     │            │   scale ┼ Float64: 1.0
     │            │      ub ┼ nothing
     │            │     rke ┴ Bool: true
     │      alpha ┼ Float64: 0.05
     │          w ┴ nothing
```

# Related

  - [`HierarchicalRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`NonOptimisationRiskRatioRiskMeasure`](@ref)
"""
@concrete struct RiskRatioRiskMeasure <: HierarchicalRiskMeasure
    "$(field_dict[:r1])"
    r1
    "$(field_dict[:r2])"
    r2
    function RiskRatioRiskMeasure(r1::OptimisationRiskMeasure, r2::OptimisationRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function RiskRatioRiskMeasure(; r1::OptimisationRiskMeasure = Variance(),
                              r2::OptimisationRiskMeasure = ConditionalValueatRisk())::RiskRatioRiskMeasure
    return RiskRatioRiskMeasure(r1, r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskRatioRiskMeasure`](@ref) by updating both constituent risk measures from the optimisation context.

Forwards all arguments to `factory` on `r1` and `r2`.

# Related

  - [`RiskRatioRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskRatioRiskMeasure, args...; kwargs...)::RiskRatioRiskMeasure
    r1 = factory(r.r1, args...; kwargs...)
    r2 = factory(r.r2, args...; kwargs...)
    return RiskRatioRiskMeasure(; r1 = r1, r2 = r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskRatioRiskMeasure`](@ref) updating both constituent risk measures from new portfolio weights `w`.

# Related

  - [`RiskRatioRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskRatioRiskMeasure, w::VecNum)::RiskRatioRiskMeasure
    return RiskRatioRiskMeasure(; r1 = factory(r.r1, w), r2 = factory(r.r2, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a non-optimisation risk ratio measure.

`NonOptimisationRiskRatioRiskMeasure` computes the ratio of two risk measures for analysis or reporting purposes. Unlike `RiskRatioRiskMeasure`, it is not intended for use as an objective or constraint in optimisation routines.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RiskRatio}(\\boldsymbol{x}) &= \\frac{r_1(\\boldsymbol{x})}{r_2(\\boldsymbol{x})}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RiskRatio}(\\boldsymbol{x})``: Risk ratio of the portfolio.
  - $(math_dict[:xret])
  - ``r_1``: First (numerator) base risk measure.
  - ``r_2``: Second (denominator) base risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NonOptimisationRiskRatioRiskMeasure(;
        r1::AbstractBaseRiskMeasure = Variance(),
        r2::AbstractBaseRiskMeasure = ConditionalValueatRisk()
    ) -> NonOptimisationRiskRatioRiskMeasure

Keywords correspond to the struct's fields.

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskRatioRiskMeasure`](@ref)
"""
@concrete struct NonOptimisationRiskRatioRiskMeasure <: NonOptimisationRiskMeasure
    "$(field_dict[:r1])"
    r1
    "$(field_dict[:r2])"
    r2
    function NonOptimisationRiskRatioRiskMeasure(r1::AbstractBaseRiskMeasure,
                                                 r2::AbstractBaseRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function NonOptimisationRiskRatioRiskMeasure(; r1::AbstractBaseRiskMeasure = Variance(),
                                             r2::AbstractBaseRiskMeasure = ConditionalValueatRisk())::NonOptimisationRiskRatioRiskMeasure
    return NonOptimisationRiskRatioRiskMeasure(r1, r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`NonOptimisationRiskRatioRiskMeasure`](@ref) by updating both constituent risk measures from the optimisation context.

Forwards all arguments to `factory` on `r1` and `r2`.

# Related

  - [`NonOptimisationRiskRatioRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(r::NonOptimisationRiskRatioRiskMeasure, args...;
                 kwargs...)::NonOptimisationRiskRatioRiskMeasure
    r1 = factory(r.r1, args...; kwargs...)
    r2 = factory(r.r2, args...; kwargs...)
    return NonOptimisationRiskRatioRiskMeasure(; r1 = r1, r2 = r2)
end

export RiskRatioRiskMeasure, NonOptimisationRiskRatioRiskMeasure
