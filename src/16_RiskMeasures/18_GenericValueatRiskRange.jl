"""
    ValueatRiskRMs = Union{<:ValueatRisk, <:ConditionalValueatRisk,
                             <:DistributionallyRobustConditionalValueatRisk,
                             <:EntropicValueatRisk, <:WorstRealisation,
                             <:RelativisticValueatRisk, <:PowerNormValueatRisk}

Union of the tail measures of the returns that [`GenericValueatRiskRange`](@ref) accepts as its loss tail and as its gain tail.

Each member measures one tail of a return series, so the range can apply it to the returns for the losses and to the negated returns for the gains. A range measure, a drawdown measure and a dispersion measure are not members, because none of them measures one tail of the returns.

# Related

  - [`ValueatRisk`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`EntropicValueatRisk`](@ref)
  - [`WorstRealisation`](@ref)
  - [`RelativisticValueatRisk`](@ref)
  - [`PowerNormValueatRisk`](@ref)
  - [`GenericValueatRiskRange`](@ref)
"""
const ValueatRiskRMs = Union{<:ValueatRisk, <:ConditionalValueatRisk,
                             <:DistributionallyRobustConditionalValueatRisk,
                             <:EntropicValueatRisk, <:WorstRealisation,
                             <:RelativisticValueatRisk, <:PowerNormValueatRisk}
"""
$(DocStringExtensions.TYPEDEF)

Measures the spread of the returns as the sum of a tail measure of the losses and a tail measure of the gains, two measures that the caller states.

The two tails can be of different kinds and can have different parameters. Each tail is a member of [`ValueatRiskRMs`](@ref). A range measure, a drawdown measure and a dispersion measure are not members, so the constructor does not accept them as a tail. With the two tails that a specific range measure builds, such as two [`ConditionalValueatRisk`](@ref) tails at the levels of a [`ConditionalValueatRiskRange`](@ref), the range has the value of that specific measure. The construction is the library's own.

# Mathematical definition

```math
\\begin{align}
\\mathrm{GenVaRRange}(\\boldsymbol{x}) &= \\rho_{\\mathrm{loss}}(\\boldsymbol{x}) + \\rho_{\\mathrm{gain}}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{GenVaRRange}(\\boldsymbol{x})``: Generic Value-at-Risk range.
  - ``\\rho_{\\mathrm{loss}}``: Tail measure of the losses, the field `loss`.
  - ``\\rho_{\\mathrm{gain}}``: Tail measure of the gains, the field `gain`.
  - $(math_dict[:xret])

$(math_dict[:negated_upper_tail])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GenericValueatRiskRange(;
        settings::RiskMeasureSettings = RiskMeasureSettings(),
        loss::ValueatRiskRMs = ConditionalValueatRisk(),
        gain::ValueatRiskRMs = ConditionalValueatRisk()
    ) -> GenericValueatRiskRange

Keywords correspond to the struct's fields.

The constructor strips the `rke` flag from both `loss` and `gain` via
[`no_risk_expr_risk_measure`](@ref), since their risk expressions are combined into the
outer `settings`-controlled expression.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `loss`: Recursively updated through [`factory`](@ref).
  - `gain`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `loss`: Recursively viewed through [`port_opt_view`](@ref).
  - `gain`: Recursively viewed through [`port_opt_view`](@ref).

# Functor

    (r::GenericValueatRiskRange)(x::VecNum)

Computes the range of a portfolio returns vector `x` as `loss(x) + gain(-x)`. A tail that needs a solver, such as [`EntropicValueatRisk`](@ref), needs its own `slv`, which [`factory`](@ref) fills from the solver of the optimiser.

A [`ValueatRisk`](@ref) tail with [`DistributionValueatRisk`](@ref) reads the portfolio weights and the moments of the prior, not a returns vector. The optimisation model reads such a tail correctly, but the functor cannot evaluate it, so the functor and [`expected_risk`](@ref) raise an `ArgumentError` for it. Use [`ValueatRiskRange`](@ref) with [`DistributionValueatRisk`](@ref) to evaluate that range.

## Validation

  - Neither `loss` nor `gain` is a [`ValueatRisk`](@ref) with [`DistributionValueatRisk`](@ref), else `ArgumentError`.

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
  - [`ValueatRiskRMs`](@ref)
  - [`ConditionalValueatRiskRange`](@ref): the same value, with two [`ConditionalValueatRisk`](@ref) tails.
  - [`PowerNormValueatRiskRange`](@ref): the same value, with two [`PowerNormValueatRisk`](@ref) tails.
  - [`range_tails`](@ref)
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
    function GenericValueatRiskRange(settings::RiskMeasureSettings, loss::ValueatRiskRMs,
                                     gain::ValueatRiskRMs)
        loss = no_risk_expr_risk_measure(loss)
        gain = no_risk_expr_risk_measure(gain)
        return new{typeof(settings), typeof(loss), typeof(gain)}(settings, loss, gain)
    end
end
function GenericValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                 loss::ValueatRiskRMs = ConditionalValueatRisk(),
                                 gain::ValueatRiskRMs = ConditionalValueatRisk())::GenericValueatRiskRange
    return GenericValueatRiskRange(settings, loss, gain)
end
# Deferrable slots — see `deferred_slots`. Both sides carry their own, so both the check and
# the derived recursion in `resolve_deferred_quantities` reach them through the two children.
deferred_slots(r::GenericValueatRiskRange) = (; loss = r.loss, gain = r.gain)
# Calibration slots — see `calibration_slots`. The container states no quantity of its own;
# it names the two children, and each child names its own slots. The functor calls each
# child directly rather than through `expected_risk`, so this declaration is the only route
# by which `assert_calibrated_slots` reaches a rule standing in a child.
calibration_slots(r::GenericValueatRiskRange) = (; loss = r.loss, gain = r.gain)
# Tail decomposition — see `range_tails`. This is the one range whose tails are given rather
# than derived: the caller states both measures, and the constructor has already stripped
# their risk-expression contribution. Their `ub` is *kept*, so a caller can bound one tail.
range_tails(r::GenericValueatRiskRange) = (; loss = r.loss, gain = r.gain)
# A tail keeps its `ub`, so a copy without bounds must clear the bound of each tail as well as
# the bound of the container. The method for a `RiskMeasure` rebuilds `settings` alone, so the
# corner solves of `NearOptimalCentering` would keep a tail bound. The tails keep `rke` and
# `scale`, as `no_bounds_risk_measure` states.
function no_bounds_risk_measure(r::GenericValueatRiskRange, flag::Any = nothing)
    return GenericValueatRiskRange(;
                                   settings = RiskMeasureSettings(; rke = r.settings.rke,
                                                                  scale = r.settings.scale),
                                   loss = no_bounds_risk_measure(r.loss, flag),
                                   gain = no_bounds_risk_measure(r.gain, flag))
end
function no_bounds_no_risk_expr_risk_measure(r::GenericValueatRiskRange,
                                             flag::Any = nothing)
    return GenericValueatRiskRange(;
                                   settings = RiskMeasureSettings(; rke = false,
                                                                  scale = one(r.settings.scale)),
                                   loss = no_bounds_risk_measure(r.loss, flag),
                                   gain = no_bounds_risk_measure(r.gain, flag))
end
function (r::GenericValueatRiskRange)(x::VecNum)
    @argcheck(!isa(r.loss, ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}) &&
              !isa(r.gain, ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}),
              ArgumentError("a ValueatRisk tail with DistributionValueatRisk reads the portfolio weights, not a returns vector, so GenericValueatRiskRange cannot evaluate it on a returns vector. Use ValueatRiskRange with DistributionValueatRisk."))
    loss = r.loss(x)
    gain = r.gain(-x)
    return loss + gain
end
risk_input_kind(::GenericValueatRiskRange) = NetReturnsInput()

export GenericValueatRiskRange
