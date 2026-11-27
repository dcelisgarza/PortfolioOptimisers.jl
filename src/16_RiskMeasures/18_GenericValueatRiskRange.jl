"""
    ValueatRiskRMs = Union{ValueatRisk, ConditionalValueatRisk,
                             DistributionallyRobustConditionalValueatRisk,
                             EntropicValueatRisk, WorstRealisation,
                             RelativisticValueatRisk, PowerNormValueatRisk}

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
const ValueatRiskRMs = Union{ValueatRisk, ConditionalValueatRisk,
                             DistributionallyRobustConditionalValueatRisk,
                             EntropicValueatRisk, WorstRealisation, RelativisticValueatRisk,
                             PowerNormValueatRisk}
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
outer `settings`-controlled expression. Each tail keeps its `ub`, which bounds that tail
alone. A [`Frontier`](@ref) in the `ub` of a tail takes its span from the value of that tail
at the two corner portfolios.

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
    (r::GenericValueatRiskRange)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)

The first method computes the range of a portfolio returns vector `x` as `loss(x) + gain(-x)`. A tail that needs a solver, such as [`EntropicValueatRisk`](@ref), needs its own `slv`, which [`factory`](@ref) fills from the solver of the optimiser.

A [`ValueatRisk`](@ref) tail with [`DistributionValueatRisk`](@ref) reads the portfolio weights and the moments of the prior, not a returns vector. The second method evaluates each tail on its own input through [`range_tail_risk`](@ref). It computes the net returns `x` of `w`, then evaluates the loss tail on `w` and `x`, and the gain tail on `-w` and `-x`. The gain tail of such a [`ValueatRisk`](@ref) reads the law of the standardised loss of `-w`, which [`gain_tail_distribution`](@ref) gives, through [`gain_tail_measure`](@ref). It is then `dot(mu, w) - zc * sqrt(dot(w, sigma, w))`, with `zc = compute_value_at_risk_cz(dist, alpha)`, which is the gain tail that the optimisation model builds and the gain tail of a [`ValueatRiskRange`](@ref). For a `dist` that is symmetric about zero, `-zc` equals the `z` of the loss tail. [`risk_input_kind`](@ref) of the range is [`WeightsReturnsFeesInput`](@ref) when a tail is such a [`ValueatRisk`](@ref), so [`expected_risk`](@ref) calls the second method. Otherwise it is [`NetReturnsInput`](@ref), and [`expected_risk`](@ref) calls the first method.

## Validation

  - First method: neither `loss` nor `gain` is a [`ValueatRisk`](@ref) with [`DistributionValueatRisk`](@ref), else `ArgumentError`.

## Arguments

  - `x::VecNum`: Portfolio returns vector.
  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees::Option{<:Fees}`: Optional [`Fees`](@ref) structure.

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
  - [`range_tail_risk`](@ref)
  - [`gain_tail_measure`](@ref)
  - [`GainTail`](@ref)
  - [`gain_tail_distribution`](@ref)
  - [`no_risk_expr_risk_measure`](@ref)
  - [`factory`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct GenericValueatRiskRange <: RangeRiskMeasure
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
range_tails(r::GenericValueatRiskRange) = RangeTails(r.loss, r.gain)
# A tail keeps its `ub`, so a copy without bounds must clear the bound of each tail as well as
# the bound of the container. The method for a `RiskMeasure` rebuilds `settings` alone, so the
# corner solves of `NearOptimalCentering` would keep a tail bound. The tails keep `rke` and
# `scale`, as `no_bounds_risk_measure` states.
function no_bounds_risk_measure(r::GenericValueatRiskRange)
    return GenericValueatRiskRange(;
                                   settings = rebuild_struct(r.settings, (; ub = nothing)),
                                   loss = no_bounds_risk_measure(r.loss),
                                   gain = no_bounds_risk_measure(r.gain))
end
function no_bounds_no_risk_expr_risk_measure(r::GenericValueatRiskRange)
    return GenericValueatRiskRange(;
                                   settings = RiskMeasureSettings(; rke = false,
                                                                  scale = one(r.settings.scale)),
                                   loss = no_bounds_risk_measure(r.loss),
                                   gain = no_bounds_risk_measure(r.gain))
end
function (r::GenericValueatRiskRange)(x::VecNum)
    @argcheck(!isa(r.loss, ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}) &&
              !isa(r.gain, ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}),
              ArgumentError("a ValueatRisk tail with DistributionValueatRisk reads the portfolio weights, not a returns vector, so GenericValueatRiskRange cannot evaluate it on a returns vector. Call the range on the weights, `r(w, X, fees)`, or use `expected_risk`."))
    loss = r.loss(x)
    gain = r.gain(-x)
    return loss + gain
end
"""
    range_tail_risk(r::RiskMeasure, w::VecNum, x::VecNum,
                    kind::RiskInputKind = risk_input_kind(r))
    range_tail_risk(r::Union{ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk},
                             ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                              <:DistributionValueatRisk}},
                    w::VecNum, x::VecNum)

Evaluate one tail of a range on the input that the tail reads.

The functor of a [`GenericValueatRiskRange`](@ref) on the weights calls this function once for each tail. It passes `w` and the net returns `x` for the loss tail, and [`gain_tail_measure`](@ref) of the gain tail, `-w` and `-x` for the gain tail. A [`GainTail`](@ref) calls it through [`gain_tail_risk`](@ref) for the gain tail of any [`RangeRiskMeasure`](@ref).

  - A tail that reads a return series, [`NetReturnsInput`](@ref), returns `r(x)`.
  - A tail that reads the weights alone, [`WeightsInput`](@ref), returns `r(w)`.
  - A [`ValueatRisk`](@ref) tail with [`DistributionValueatRisk`](@ref), [`WeightsReturnsFeesInput`](@ref), returns `r(w)`. It reads the moments of the prior against the weights. A [`ValueatRiskRange`](@ref) tail with that formulation does the same.
  - Any other [`WeightsReturnsFeesInput`](@ref) tail throws an `ArgumentError`. Its functor reads the asset returns and the fees, which this function does not take, so the value of a gain tail on the negated portfolio is not defined. The tails of a [`GenericValueatRiskRange`](@ref) never meet this method. [`gain_tail_risk`](@ref) values a [`SymmetricLowOrderMoment`](@ref) and a range that reads the weights before this method, so a [`GainTail`](@ref) meets it only for a measure of a user that declares [`supports_gain_tail`](@ref) and no method of [`gain_tail_risk`](@ref).

The gain tail of a returns measure reads `-x`, not the net returns of `-w`. A fee does not change sign with the weights, so the two series are different when the portfolio pays a fee.

# Arguments

  - `r`: Tail measure.
  - `w::VecNum`: Portfolio weights vector `assets × 1`, negated for the gain tail.
  - `x::VecNum`: Net portfolio returns vector `observations × 1`, negated for the gain tail.
  - `kind::RiskInputKind`: Input kind of the tail, which selects the method.

# Validation

  - A [`WeightsReturnsFeesInput`](@ref) tail that is not a [`ValueatRisk`](@ref) or a [`ValueatRiskRange`](@ref) with [`DistributionValueatRisk`](@ref) throws an `ArgumentError`.

# Returns

  - `rk::Number`: The value of the tail.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`GainTail`](@ref)
  - [`gain_tail_measure`](@ref)
  - [`risk_input_kind`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
function range_tail_risk(r::RiskMeasure, ::VecNum, ::VecNum,
                         ::WeightsReturnsFeesInput = risk_input_kind(r))
    return throw(ArgumentError("`$(typeof(r))` is a `WeightsReturnsFeesInput` risk measure, so a range cannot value it as a tail. Its functor reads the asset returns and the fees, and the gain tail of a range reads the negated weights and the negated net returns. Use a tail that reads the net returns or the weights alone, a `ValueatRisk` or a `ValueatRiskRange` with `DistributionValueatRisk`, a `LowOrderMoment` that is symmetric, or a range that reads the weights. For a measure of your own, declare a method of `PortfolioOptimisers.gain_tail_risk` that gives its value on the negated portfolio."))
end
function range_tail_risk(r::RiskMeasure, ::VecNum, x::VecNum, ::NetReturnsInput)
    return r(x)
end
function range_tail_risk(r::RiskMeasure, w::VecNum, ::VecNum, ::WeightsInput)
    return r(w)
end
function range_tail_risk(r::Union{ValueatRisk{<:Any, <:Any, <:Any,
                                              <:DistributionValueatRisk},
                                  ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:DistributionValueatRisk}}, w::VecNum,
                         ::VecNum)
    return r(w)
end
"""
    gain_tail_measure(r::RiskMeasure)
    gain_tail_measure(r::Union{ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk},
                               ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                <:DistributionValueatRisk}})

Return the measure that the gain tail of a range evaluates on the negated portfolio.

  - The method for a [`RiskMeasure`](@ref) returns `r` unchanged. A tail that reads a return series reads the negated returns, and a tail that reads the weights alone reads the negated weights.
  - A [`ValueatRisk`](@ref) tail with [`DistributionValueatRisk`](@ref) is rebuilt with the law that [`gain_tail_distribution`](@ref) gives, the law of the standardised loss of `-w`. On `-w`, the rebuilt tail then equals the gain tail of a [`ValueatRiskRange`](@ref) with the same `dist`, also for a `Normal` with a nonzero mean. A law that is symmetric about zero is its own reflection, so the method returns `r` itself. A [`ValueatRiskRange`](@ref) tail with that formulation is rebuilt in the same way.

# Arguments

  - `r`: Gain tail of the range.

# Returns

  - `r`: The measure to evaluate on the negated portfolio.

# Related

  - [`GenericValueatRiskRange`](@ref)
  - [`GainTail`](@ref)
  - [`range_tail_risk`](@ref)
  - [`gain_tail_distribution`](@ref)
  - [`DistributionValueatRisk`](@ref)
"""
gain_tail_measure(r::RiskMeasure) = r
function gain_tail_measure(r::Union{ValueatRisk{<:Any, <:Any, <:Any,
                                                <:DistributionValueatRisk},
                                    ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                     <:DistributionValueatRisk}})
    alg = rebuild_with_slots(r.alg, (; dist = gain_tail_distribution(r.alg.dist)))
    return rebuild_with_slots(r, (; alg = alg))
end
"""
    const SymmetricLowOrderMoment = LowOrderMoment{<:Any, <:Any, <:Option{<:VecNum},
                                                   <:Union{MeanAbsoluteDeviation,
                                                           SecondMoment{<:Any, <:FullMoment},
                                                           EvenMoment{<:Any, <:Any, <:FullMoment}}}

Alias for a [`LowOrderMoment`](@ref) that has one value on a portfolio and on the negated portfolio.

A moment of this kind reads the deviations of both signs alike, and its target, the mean of the portfolio, changes sign with the portfolio. A moment of the deviations below the target, and a moment about a target that holds a scalar, have two values.

# Related

  - [`LowOrderMoment`](@ref)
  - [`MeanAbsoluteDeviation`](@ref)
  - [`SecondMoment`](@ref)
  - [`EvenMoment`](@ref)
  - [`FullMoment`](@ref)
  - [`supports_gain_tail`](@ref)
  - [`gain_tail_risk`](@ref)
"""
const SymmetricLowOrderMoment = LowOrderMoment{<:Any, <:Any, <:Option{<:VecNum},
                                               <:Union{MeanAbsoluteDeviation,
                                                       SecondMoment{<:Any, <:FullMoment},
                                                       EvenMoment{<:Any, <:Any,
                                                                  <:FullMoment}}}
"""
    gain_tail_risk(r::RiskMeasure, w::VecNum, X::MatNum, fees::Option{<:Fees})
    gain_tail_risk(r::RiskMeasure, w, X, fees, kind::RiskInputKind)
    gain_tail_risk(r::SymmetricLowOrderMoment, w::VecNum, X::MatNum, fees::Option{<:Fees},
                   kind::WeightsReturnsFeesInput)
    gain_tail_risk(r::RangeRiskMeasure, w, X, fees, kind::WeightsReturnsFeesInput)
    gain_tail_risk(r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk},
                   w::VecNum, X::MatNum, fees::Option{<:Fees},
                   kind::WeightsReturnsFeesInput)

Evaluate the measure `r` as the gain tail of a range, on the portfolio `w`.

  - The method with four arguments gives `r` and [`risk_input_kind`](@ref) of `r` to the method with five.
  - The method for a [`RiskMeasure`](@ref) evaluates [`gain_tail_measure`](@ref) of `r` on `-w` and on the negated net returns, with [`range_tail_risk`](@ref).
  - A [`SymmetricLowOrderMoment`](@ref) has one value on both sides, so the method evaluates `r` on `w`.
  - A range that reads the weights exchanges the sides of its tails, as the builder of a [`RangeRiskMeasure`](@ref) does. The method returns the sum of the value of its loss tail as a gain tail and of the value of its gain tail on the portfolio.
  - A [`ValueatRiskRange`](@ref) with [`DistributionValueatRisk`](@ref) declares no tails, so the method evaluates [`gain_tail_measure`](@ref) of `r` on `-w`.

A measure of your own that reads the weights, the returns and the fees, and that declares [`supports_gain_tail`](@ref), also declares a method of this function, `gain_tail_risk(r::MyRisk, w, X, fees, ::WeightsReturnsFeesInput)`. The methods for a [`RiskMeasure`](@ref) and for a [`RangeRiskMeasure`](@ref) with five arguments type only `r` and `kind`, so such a method needs no other type to be more specific. A fee keeps its sign when the weights change sign, so no generic value of such a measure on the negated portfolio exists. A range of your own with a formulation that fuses and declares no [`range_tails`](@ref) is such a measure, as the `# Interfaces` section of [`RangeRiskMeasure`](@ref) states.

# Arguments

  - $(arg_dict[:r])
  - `w::VecNum`: Portfolio weights vector `assets × 1`, not negated.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees::Option{<:Fees}`: The fees that the net returns pay, or `nothing`.
  - `kind::RiskInputKind`: Input kind of `r`, which selects the method.

# Returns

  - `rk::Number`: The value of the gain tail.

# Related

  - [`GainTail`](@ref)
  - [`RangeTails`](@ref)
  - [`range_tails`](@ref)
  - [`gain_tail_measure`](@ref)
  - [`range_tail_risk`](@ref)
  - [`risk_input_kind`](@ref)
"""
function gain_tail_risk(r::RiskMeasure, w::VecNum, X::MatNum, fees::Option{<:Fees})
    return gain_tail_risk(r, w, X, fees, risk_input_kind(r))
end
# The two methods that a method of a user meets type `r` and the kind alone, so that a method
# of a user with untyped `w`, `X` and `fees` is more specific and not ambiguous.
function gain_tail_risk(r::RiskMeasure, w, X, fees, ::RiskInputKind)
    return range_tail_risk(gain_tail_measure(r), -w, -calc_net_returns(w, X, fees))
end
function gain_tail_risk(r::SymmetricLowOrderMoment, w::VecNum, X::MatNum,
                        fees::Option{<:Fees}, ::WeightsReturnsFeesInput)
    return r(w, X, fees)
end
function gain_tail_risk(r::RangeRiskMeasure, w, X, fees, ::WeightsReturnsFeesInput)
    (; loss, gain) = range_tails(r)
    return gain_tail_risk(loss, w, X, fees) + expected_risk(gain, w, X, fees)
end
function gain_tail_risk(r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                            <:DistributionValueatRisk}, w::VecNum, ::MatNum,
                        ::Option{<:Fees}, ::WeightsReturnsFeesInput)
    return gain_tail_measure(r)(-w)
end
"""
$(DocStringExtensions.TYPEDEF)

Measures the gain tail of a range alone: the tail measure `r` on the negated portfolio.

The builders of a [`RangeRiskMeasure`](@ref) record a `GainTail` as the owner of the [`Frontier`](@ref) that the gain tail holds, as [`set_risk_frontier_owner!`](@ref) states. The tail measure itself is no owner, because on the portfolio it gives the value of a loss tail. The loss tail reads the portfolio, so it is its own owner.

# Mathematical definition

```math
\\begin{align}
\\mathrm{GainTail}(\\boldsymbol{x}) &= \\rho_{\\mathrm{gain}}(-\\boldsymbol{x})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{GainTail}(\\boldsymbol{x})``: Value of the gain tail.
  - ``\\rho_{\\mathrm{gain}}``: Tail measure of the gains, the field `r`.
  - $(math_dict[:xret])

$(math_dict[:negated_upper_tail])

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GainTail(r::RiskMeasure) -> GainTail
    GainTail(; r::RiskMeasure) -> GainTail

Keywords correspond to the struct's fields.

## Propagated parameters

[`factory`](@ref) passes its arguments on to these `@fprop` fields:

  - `r`: Recursively updated through [`factory`](@ref).

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `r`: Recursively viewed through [`port_opt_view`](@ref).

# Functor

    (r::GainTail)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)

Evaluates the tail on the negated portfolio with [`gain_tail_risk`](@ref). The value of the range is the sum of the value of the loss tail and of this value. A [`WeightsReturnsFeesInput`](@ref) tail that no method of [`gain_tail_risk`](@ref) values throws an `ArgumentError`, as [`range_tail_risk`](@ref) states.

## Arguments

  - `w::VecNum`: Portfolio weights vector `assets × 1`.
  - `X::MatNum`: Returns matrix `observations × assets`.
  - `fees::Option{<:Fees}`: Optional [`Fees`](@ref) structure.

# Examples

```jldoctest
julia> r = PortfolioOptimisers.GainTail(WorstRealisation());

julia> r([1.0], [-0.1; 0.25; 0.5;;])
0.5
```

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`RangeRiskMeasure`](@ref)
  - [`RangeTails`](@ref)
  - [`GenericValueatRiskRange`](@ref)
  - [`set_risk_frontier_owner!`](@ref)
  - [`gain_tail_risk`](@ref)
  - [`gain_tail_measure`](@ref)
  - [`range_tail_risk`](@ref)
"""
@propagatable @concrete struct GainTail <: NonOptimisationRiskMeasure
    """
    The point measure of the gains, the `gain` of a [`RangeTails`](@ref).
    """
    @fprop @vprop r
    function GainTail(r::RiskMeasure)
        return new{typeof(r)}(r)
    end
end
function GainTail(; r::RiskMeasure)::GainTail
    return GainTail(r)
end
# Deferrable and calibration slots — see `deferred_slots` and `calibration_slots`. The functor
# calls the tail directly, so these declarations are the only route to the slots of the tail.
deferred_slots(r::GainTail) = (; r = r.r)
calibration_slots(r::GainTail) = (; r = r.r)
# Evaluation entry points. The value reads the weights and the returns, and never a series alone.
risk_input_kind(::GainTail) = WeightsReturnsFeesInput()
supports_precomputed_returns(::GainTail)::Bool = false
function (r::GainTail)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    return gain_tail_risk(r.r, w, X, fees)
end
"""
    supports_gain_tail(r::RiskMeasure) -> Bool
    supports_gain_tail(r::RangeRiskMeasure) -> Bool
    supports_gain_tail(r::Union{ValueatRiskRMs, OrderedWeightsArray, NoRisk}) -> Bool
    supports_gain_tail(r::Union{Variance, StandardDeviation, UncertaintySetVariance, Kurtosis,
                                BrownianDistanceVariance, SymmetricLowOrderMoment}) -> Bool
    supports_gain_tail(r::Union{Range, OrderedWeightsArrayRange, ValueatRiskRange}) -> Bool

Return whether the risk measure `r` can be the gain tail of a [`RangeRiskMeasure`](@ref).

A gain tail measures the negated portfolio. Its value reads the negated net returns, or the negated weights, as its [`risk_input_kind`](@ref) states, and its builder must build that value under `loss = false`. A builder that ignores the keyword builds the loss tail a second time, and the model then optimises a value that is not the value of the range. So [`RangeTails`](@ref) refuses a gain tail whose answer is `false`.

  - A measure answers `true` when its builder reads the side of its `loss` keyword. The tail measures of the returns, such as [`ConditionalValueatRisk`](@ref), [`WorstRealisation`](@ref) and [`OrderedWeightsArray`](@ref), do so.
  - A measure answers `true` when its value is the same on the negated portfolio, so that its builder needs no side. [`Variance`](@ref), [`StandardDeviation`](@ref), [`UncertaintySetVariance`](@ref), [`Kurtosis`](@ref), [`BrownianDistanceVariance`](@ref), [`Range`](@ref) and a [`SymmetricLowOrderMoment`](@ref) do so. A measure that reads a matrix reads the same matrix on both sides, so a lower semi-moment in the matrix keeps its side.
  - [`OrderedWeightsArrayRange`](@ref) and [`ValueatRiskRange`](@ref) answer `true`. A formulation of them that fuses its two tails builds the side that the keyword states, and a formulation that decomposes has tails with a side.
  - Any other range that decomposes answers for its loss tail, which it builds on the negated portfolio when the range is a gain tail. Its gain tail is then built on the portfolio, which each measure supports.
  - Every other measure answers `false`, such as a drawdown measure, a [`LowOrderMoment`](@ref) of any other kind, [`NegativeSkewness`](@ref), a tracking measure and [`TurnoverRiskMeasure`](@ref).

# Arguments

  - $(arg_dict[:r])

# Returns

  - `flag::Bool`: Whether `r` can be a gain tail.

# Related

  - [`RangeRiskMeasure`](@ref)
  - [`RangeTails`](@ref)
  - [`GainTail`](@ref)
  - [`gain_tail_measure`](@ref)
  - [`range_tails`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
supports_gain_tail(::RiskMeasure)::Bool = false
function supports_gain_tail(r::RangeRiskMeasure)::Bool
    return supports_gain_tail(range_tails(r).loss)
end
# The builder of each of these measures reads the side of its `loss` keyword.
function supports_gain_tail(::Union{ValueatRiskRMs, OrderedWeightsArray, NoRisk})::Bool
    return true
end
# The value of each of these measures is the same on the negated portfolio.
function supports_gain_tail(::Union{Variance, StandardDeviation, UncertaintySetVariance,
                                    Kurtosis, BrownianDistanceVariance,
                                    SymmetricLowOrderMoment})::Bool
    return true
end
# A formulation that fuses declares no `range_tails`, so these ranges answer for their type.
function supports_gain_tail(::Union{Range, OrderedWeightsArrayRange, ValueatRiskRange})::Bool
    return true
end
# A range with a tail that reads the weights reads the weights, the returns and the fees. The
# loss tail reads `w` and `x`, and the gain tail reads `-w` and `-x`, as the model builds them.
# The gain tail of a `DistributionValueatRisk` reads the law of the negated standardised loss.
function (r::GenericValueatRiskRange)(w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    loss = range_tail_risk(r.loss, w, x)
    gain = range_tail_risk(gain_tail_measure(r.gain), -w, -x)
    return loss + gain
end
# Expected-risk input kind — see `risk_input_kind`. The kind follows the tails: a range whose
# tails both read a return series reads the net returns, and a range with a `ValueatRisk` tail
# with `DistributionValueatRisk` reads the weights, the returns and the fees.
risk_input_kind(::GenericValueatRiskRange) = NetReturnsInput()
function risk_input_kind(::Union{GenericValueatRiskRange{<:Any,
                                                         <:ValueatRisk{<:Any, <:Any, <:Any,
                                                                       <:DistributionValueatRisk},
                                                         <:Any},
                                 GenericValueatRiskRange{<:Any, <:Any,
                                                         <:ValueatRisk{<:Any, <:Any, <:Any,
                                                                       <:DistributionValueatRisk}}})
    return WeightsReturnsFeesInput()
end
# The range reads a precomputed return series only when both tails do.
function supports_precomputed_returns(r::GenericValueatRiskRange)
    return all(supports_precomputed_returns, (r.loss, r.gain))
end

export GenericValueatRiskRange
# The verbs the `# Interfaces` sections of `RiskMeasure` and `RangeRiskMeasure` name (ADR 0154).
public supports_gain_tail, gain_tail_risk
