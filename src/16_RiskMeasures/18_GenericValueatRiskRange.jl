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
    range_tail_risk(r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk},
                    w::VecNum, x::VecNum)

Evaluate one tail of a range on the input that the tail reads.

The functor of a [`GenericValueatRiskRange`](@ref) on the weights calls this function once for each tail. It passes `w` and the net returns `x` for the loss tail, and [`gain_tail_measure`](@ref) of the gain tail, `-w` and `-x` for the gain tail. A [`GainTail`](@ref) calls it for the gain tail of any [`RangeRiskMeasure`](@ref).

  - A tail that reads a return series, [`NetReturnsInput`](@ref), returns `r(x)`.
  - A tail that reads the weights alone, [`WeightsInput`](@ref), returns `r(w)`.
  - A [`ValueatRisk`](@ref) tail with [`DistributionValueatRisk`](@ref), [`WeightsReturnsFeesInput`](@ref), returns `r(w)`. It reads the moments of the prior against the weights.
  - Any other [`WeightsReturnsFeesInput`](@ref) tail throws an `ArgumentError`. Its functor reads the asset returns and the fees, which this function does not take, so the value of a gain tail on the negated portfolio is not defined. The tails of a [`GenericValueatRiskRange`](@ref) never meet this method. A [`GainTail`](@ref) of a range of a user meets it when [`rebuild_risk_frontier`](@ref) values the corners of a [`Frontier`](@ref) on the gain tail.

The gain tail of a returns measure reads `-x`, not the net returns of `-w`. A fee does not change sign with the weights, so the two series are different when the portfolio pays a fee.

# Arguments

  - `r`: Tail measure.
  - `w::VecNum`: Portfolio weights vector `assets × 1`, negated for the gain tail.
  - `x::VecNum`: Net portfolio returns vector `observations × 1`, negated for the gain tail.
  - `kind::RiskInputKind`: Input kind of the tail, which selects the method.

# Validation

  - A [`WeightsReturnsFeesInput`](@ref) tail that is not a [`ValueatRisk`](@ref) with [`DistributionValueatRisk`](@ref) throws an `ArgumentError`.

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
    return throw(ArgumentError("`$(typeof(r))` is a `WeightsReturnsFeesInput` risk measure, so a range cannot value it as a tail. Its functor reads the asset returns and the fees, and the gain tail of a range reads the negated weights and the negated net returns. Use a tail that reads the net returns or the weights alone, or a `ValueatRisk` with `DistributionValueatRisk`."))
end
function range_tail_risk(r::RiskMeasure, ::VecNum, x::VecNum, ::NetReturnsInput)
    return r(x)
end
function range_tail_risk(r::RiskMeasure, w::VecNum, ::VecNum, ::WeightsInput)
    return r(w)
end
function range_tail_risk(r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk},
                         w::VecNum, ::VecNum)
    return r(w)
end
"""
    gain_tail_measure(r::RiskMeasure)
    gain_tail_measure(r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk})

Return the measure that the gain tail of a range evaluates on the negated portfolio.

  - The method for a [`RiskMeasure`](@ref) returns `r` unchanged. A tail that reads a return series reads the negated returns, and a tail that reads the weights alone reads the negated weights.
  - A [`ValueatRisk`](@ref) tail with [`DistributionValueatRisk`](@ref) is rebuilt with the law that [`gain_tail_distribution`](@ref) gives, the law of the standardised loss of `-w`. On `-w`, the rebuilt tail then equals the gain tail of a [`ValueatRiskRange`](@ref) with the same `dist`, also for a `Normal` with a nonzero mean. A law that is symmetric about zero is its own reflection, so the method returns `r` itself.

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
function gain_tail_measure(r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk})
    alg = rebuild_with_slots(r.alg, (; dist = gain_tail_distribution(r.alg.dist)))
    return rebuild_with_slots(r, (; alg = alg))
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

Computes the net returns `x` of `w`, and evaluates [`gain_tail_measure`](@ref) of the tail on `-w` and `-x` with [`range_tail_risk`](@ref). The value of the range is the sum of the value of the loss tail and of this value. A [`WeightsReturnsFeesInput`](@ref) tail that is not a [`ValueatRisk`](@ref) with [`DistributionValueatRisk`](@ref) throws an `ArgumentError`, as [`range_tail_risk`](@ref) states.

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
    return range_tail_risk(gain_tail_measure(r.r), -w, -calc_net_returns(w, X, fees))
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
