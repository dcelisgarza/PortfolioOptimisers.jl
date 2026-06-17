"""
$(DocStringExtensions.TYPEDEF)

Represents a generic Value-at-Risk range risk measure that combines any pair of XatRisk-type
measures applied to the loss and gain sides of the return distribution.

`GenericValueatRiskRange` evaluates a loss-side XatRisk measure on the portfolio returns and
a gain-side XatRisk measure on the negated portfolio returns, then sums the two to produce a
symmetric tail-spread risk measure.

# Mathematical definition

```math
\\begin{align}
\\mathrm{GenVaRRange}(\\boldsymbol{x}) &= \\rho_{\\mathrm{loss}}(\\boldsymbol{x}) + \\rho_{\\mathrm{gain}}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{GenVaRRange}(\\boldsymbol{x})``: Generic Value-at-Risk range.
  - ``\\rho_{\\mathrm{loss}}``: Loss-side XatRisk risk measure.
  - ``\\rho_{\\mathrm{gain}}``: Gain-side XatRisk risk measure.
  - $(math_dict[:xret])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GenericValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        loss::Union{<:ValueatRisk, <:ConditionalValueatRisk,
                    <:DistributionallyRobustConditionalValueatRisk,
                    <:EntropicValueatRisk, <:RelativisticValueatRisk,
                    <:PowerNormValueatRisk} = ConditionalValueatRisk(),
        gain::Union{<:ValueatRisk, <:ConditionalValueatRisk,
                    <:DistributionallyRobustConditionalValueatRisk,
                    <:EntropicValueatRisk, <:RelativisticValueatRisk,
                    <:PowerNormValueatRisk} = ConditionalValueatRisk()
    ) -> GenericValueatRiskRange

Keywords correspond to the struct's fields.

The constructor strips the `rke` flag from both `loss` and `gain` via
[`no_risk_expr_risk_measure`](@ref), since their risk expressions are combined into the
outer `settings`-controlled expression.

## Propagated parameters

When [`factory`](@ref) is called on this type, the following `@fprop`-tagged fields are
automatically propagated:

  - `loss`: Recursively updated via [`factory`](@ref).
  - `gain`: Recursively updated via [`factory`](@ref).

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields
are automatically subset to the selected indices:

  - `loss`: Recursively viewed via [`port_opt_view`](@ref).
  - `gain`: Recursively viewed via [`port_opt_view`](@ref).

# Functor

    (r::GenericValueatRiskRange)(x::VecNum)

Computes the GenericValueatRiskRange of a portfolio returns vector `x`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.

# Examples

```jldoctest
julia> GenericValueatRiskRange()
GenericValueatRiskRange
  settings ┼ RiskMeasureSettings
           │   scale ┼ Float64: 1.0
           │      ub ┼ nothing
           │     rke ┴ Bool: true
      loss ┼ ConditionalValueatRisk
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │      alpha ┼ Float64: 0.05
           │          w ┴ nothing
      gain ┼ ConditionalValueatRisk
           │   settings ┼ RiskMeasureSettings
           │            │   scale ┼ Float64: 1.0
           │            │      ub ┼ nothing
           │            │     rke ┴ Bool: false
           │      alpha ┼ Float64: 0.05
           │          w ┴ nothing
```

# Related

  - [`RiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`ValueatRisk`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`PowerNormValueatRisk`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct GenericValueatRiskRange <: RiskMeasure
    """
    $(field_dict[:settings_rm])
    """
    settings
    """
    $(field_dict[:loss_rm])
    """
    @fprop @vprop loss
    """
    $(field_dict[:gain_rm])
    """
    @fprop @vprop gain
    function GenericValueatRiskRange(settings::RiskMeasureSettings,
                                     loss::Union{<:ValueatRisk, <:ConditionalValueatRisk,
                                                 <:DistributionallyRobustConditionalValueatRisk,
                                                 <:EntropicValueatRisk,
                                                 <:RelativisticValueatRisk,
                                                 <:WorstRealisation,
                                                 <:PowerNormValueatRisk},
                                     gain::Union{<:ValueatRisk, <:ConditionalValueatRisk,
                                                 <:DistributionallyRobustConditionalValueatRisk,
                                                 <:EntropicValueatRisk,
                                                 <:RelativisticValueatRisk,
                                                 <:WorstRealisation,
                                                 <:PowerNormValueatRisk})
        loss = no_risk_expr_risk_measure(loss)
        gain = no_risk_expr_risk_measure(gain)
        return new{typeof(settings), typeof(loss), typeof(gain)}(settings, loss, gain)
    end
end
function GenericValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 loss::Union{<:ValueatRisk, <:ConditionalValueatRisk,
                                             <:DistributionallyRobustConditionalValueatRisk,
                                             <:EntropicValueatRisk,
                                             <:RelativisticValueatRisk, <:WorstRealisation,
                                             <:PowerNormValueatRisk} = ConditionalValueatRisk(),
                                 gain::Union{<:ValueatRisk, <:ConditionalValueatRisk,
                                             <:DistributionallyRobustConditionalValueatRisk,
                                             <:EntropicValueatRisk,
                                             <:RelativisticValueatRisk, <:WorstRealisation,
                                             <:PowerNormValueatRisk} = ConditionalValueatRisk())::GenericValueatRiskRange
    return GenericValueatRiskRange(settings, loss, gain)
end
function (r::GenericValueatRiskRange)(x::VecNum)
    loss = r.loss(x)
    gain = r.gain(-x)
    return loss + gain
end
risk_input_kind(::GenericValueatRiskRange) = NetReturnsInput()

export GenericValueatRiskRange
