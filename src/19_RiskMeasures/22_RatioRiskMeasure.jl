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

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `r1`: Recursively updated via [`factory`](@ref).
  - `r2`: Recursively updated via [`factory`](@ref).

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
@propagatable @concrete struct RiskRatio <: HierarchicalRiskMeasure
    """
    $(field_dict[:r1])
    """
    @fprop r1
    """
    $(field_dict[:r2])
    """
    @fprop r2
    function RiskRatio(r1::OptimisationRiskMeasure, r2::OptimisationRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function RiskRatio(; r1::OptimisationRiskMeasure = Variance(),
                   r2::OptimisationRiskMeasure = ConditionalValueatRisk())::RiskRatio
    return RiskRatio(r1, r2)
end
# Deferrable slots — see `deferred_slots`. The two measures carry their own, so both the check
# and the derived recursion in `resolve_deferred_quantities` reach them through the children.
deferred_slots(r::RiskRatio) = (; r1 = r.r1, r2 = r.r2)
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
        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
        r1::BaseRM_VecBaseRM = Variance(),
        sca1::Scalariser = SumScalariser(),
        r2::BaseRM_VecBaseRM = ConditionalValueatRisk(),
        sca2::Scalariser = SumScalariser()
    ) -> NonOptimisationRiskRatio

Keywords correspond to the struct's fields.

## Multiplicity

Each axis takes one risk measure or a vector of them, and each carries **its own** scalariser. This is the only type in the family with two independent risk vectors, so `sca1` governs `r1` and `sca2` governs `r2`. A scalariser sits immediately after the field it governs.

Both fields beat a caller's `sca` keyword. A `sca` passed at the call site flows no further than this type, so a figure reported from a `NonOptimisationRiskRatio` is always the pair the type names. The two axes are independent: `sca1` and `sca2` need not agree.

## Validation

  - If `r1` is a vector: `!isempty(r1)`.
  - If `r2` is a vector: `!isempty(r2)`.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are automatically propagated:

  - `r1`: Recursively updated via [`factory`](@ref).
  - `r2`: Recursively updated via [`factory`](@ref).

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskRatio`](@ref)
"""
@propagatable @concrete struct NonOptimisationRiskRatio <: NonOptimisationRiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:r1_vec])
    """
    @fprop r1
    """
    $(field_dict[:sca_r1])
    """
    sca1
    """
    $(field_dict[:r2_vec])
    """
    @fprop r2
    """
    $(field_dict[:sca_r2])
    """
    sca2
    function NonOptimisationRiskRatio(settings::HierarchicalRiskMeasureSettings,
                                      r1::BaseRM_VecBaseRM, sca1::Scalariser,
                                      r2::BaseRM_VecBaseRM, sca2::Scalariser)
        if isa(r1, VecBaseRM)
            @argcheck(!isempty(r1), IsEmptyError("r1 cannot be empty"))
        end
        if isa(r2, VecBaseRM)
            @argcheck(!isempty(r2), IsEmptyError("r2 cannot be empty"))
        end
        return new{typeof(settings), typeof(r1), typeof(sca1), typeof(r2), typeof(sca2)}(settings,
                                                                                         r1,
                                                                                         sca1,
                                                                                         r2,
                                                                                         sca2)
    end
end
function NonOptimisationRiskRatio(;
                                  settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                  r1::BaseRM_VecBaseRM = Variance(),
                                  sca1::Scalariser = SumScalariser(),
                                  r2::BaseRM_VecBaseRM = ConditionalValueatRisk(),
                                  sca2::Scalariser = SumScalariser())::NonOptimisationRiskRatio
    return NonOptimisationRiskRatio(settings, r1, sca1, r2, sca2)
end
# Deferrable slots — see `deferred_slots`. Each side is one measure or a vector of them, and
# the derived recursion resolves a vector element by element.
deferred_slots(r::NonOptimisationRiskRatio) = (; r1 = r.r1, r2 = r.r2)

export RiskRatio, NonOptimisationRiskRatio
