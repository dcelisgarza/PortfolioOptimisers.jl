"""
$(DocStringExtensions.TYPEDEF)

Represents a risk ratio risk measure for hierarchical portfolio optimisation.

`RiskRatio` computes the ratio of two risk measures, enabling the construction of risk-adjusted performance metrics for use in hierarchical optimisation routines.

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

    RiskRatio(;
        r1::OptimisationRiskMeasure = Variance(),
        r2::OptimisationRiskMeasure = ConditionalValueatRisk()
    ) -> RiskRatio

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> RiskRatio()
RiskRatio
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
  - [`NonOptimisationRiskRatio`](@ref)
"""
@concrete struct RiskRatio <: HierarchicalRiskMeasure
    """
    $(field_dict[:r1])
    """
    r1
    """
    $(field_dict[:r2])
    """
    r2
    function RiskRatio(r1::OptimisationRiskMeasure, r2::OptimisationRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function RiskRatio(; r1::OptimisationRiskMeasure = Variance(),
                   r2::OptimisationRiskMeasure = ConditionalValueatRisk())::RiskRatio
    return RiskRatio(r1, r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskRatio`](@ref) by updating both constituent risk measures from the optimisation context.

Forwards all arguments to `factory` on `r1` and `r2`.

# Related

  - [`RiskRatio`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskRatio, args...; kwargs...)::RiskRatio
    r1 = factory(r.r1, args...; kwargs...)
    r2 = factory(r.r2, args...; kwargs...)
    return RiskRatio(; r1 = r1, r2 = r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`RiskRatio`](@ref) updating both constituent risk measures from new portfolio weights `w`.

# Related

  - [`RiskRatio`](@ref)
  - [`factory`](@ref)
"""
function factory(r::RiskRatio, w::VecNum)::RiskRatio
    return RiskRatio(; r1 = factory(r.r1, w), r2 = factory(r.r2, w))
end
"""
$(DocStringExtensions.TYPEDEF)

Represents a non-optimisation risk ratio measure.

`NonOptimisationRiskRatio` computes the ratio of two risk measures for analysis or reporting purposes. Unlike `RiskRatio`, it is not intended for use as an objective or constraint in optimisation routines.

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

    NonOptimisationRiskRatio(;
        r1::AbstractBaseRiskMeasure = Variance(),
        r2::AbstractBaseRiskMeasure = ConditionalValueatRisk()
    ) -> NonOptimisationRiskRatio

Keywords correspond to the struct's fields.

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskRatio`](@ref)
"""
@concrete struct NonOptimisationRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:r1])
    """
    r1
    """
    $(field_dict[:r2])
    """
    r2
    function NonOptimisationRiskRatio(r1::AbstractBaseRiskMeasure,
                                      r2::AbstractBaseRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function NonOptimisationRiskRatio(; r1::AbstractBaseRiskMeasure = Variance(),
                                  r2::AbstractBaseRiskMeasure = ConditionalValueatRisk())::NonOptimisationRiskRatio
    return NonOptimisationRiskRatio(r1, r2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Create an instance of [`NonOptimisationRiskRatio`](@ref) by updating both constituent risk measures from the optimisation context.

Forwards all arguments to `factory` on `r1` and `r2`.

# Related

  - [`NonOptimisationRiskRatio`](@ref)
  - [`factory`](@ref)
"""
function factory(r::NonOptimisationRiskRatio, args...; kwargs...)::NonOptimisationRiskRatio
    r1 = factory(r.r1, args...; kwargs...)
    r2 = factory(r.r2, args...; kwargs...)
    return NonOptimisationRiskRatio(; r1 = r1, r2 = r2)
end

export RiskRatio, NonOptimisationRiskRatio
