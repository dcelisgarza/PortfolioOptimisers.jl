"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all risk measure estimators.

Defines the interface for risk measure types, which quantify portfolio risk using various statistical or econometric methods. All concrete risk measure types should subtype `AbstractBaseRiskMeasure` to ensure consistency and composability within the optimisation framework.

All concrete risk measures can be used as functors (callable structs) to compute their associated risk quantity.

# Related

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type AbstractBaseRiskMeasure <: AbstractEstimator end
"""
    needs_previous_weights(r::AbstractBaseRiskMeasure) -> Bool
    needs_previous_weights(r::VecBaseRM) -> Bool

Return whether risk measure `r` requires previous portfolio weights as input.

The default returns `false`. Override to `true` for risk measures such as turnover or tracking constraints that depend on prior weights. The `VecBaseRM` overload returns `true` if any element returns `true`.

# Returns

  - `Bool`: `true` if prior weights are required; `false` otherwise.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`VecBaseRM`](@ref)
"""
function needs_previous_weights(::AbstractBaseRiskMeasure)::Bool
    return false
end
"""
    bigger_is_better(r::AbstractBaseRiskMeasure) -> Bool

Return whether a larger value of risk measure `r` is preferred over a smaller one.

The default implementation returns `false` (lower risk is better) for all [`AbstractBaseRiskMeasure`](@ref) subtypes. Ratio-based or return-like measures that should be maximised may override this method to return `true`.

# Returns

  - `Bool`: `true` if a higher value of `r` is preferred; `false` otherwise.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
"""
function bigger_is_better(::AbstractBaseRiskMeasure)::Bool
    return false
end
"""
    const VecBaseRM = AbstractVector{<:AbstractBaseRiskMeasure}

Alias for an abstract vector of [`AbstractBaseRiskMeasure`](@ref) elements.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`VecOptRM`](@ref)
  - [`VecRM`](@ref)
"""
const VecBaseRM = AbstractVector{<:AbstractBaseRiskMeasure}
function needs_previous_weights(r::VecBaseRM)::Bool
    return any(needs_previous_weights.(r))
end
"""
    const BaseRM_VecBaseRM = Union{<:AbstractBaseRiskMeasure, <:VecBaseRM}

Argument bound for every value-level reader of a risk measure: one measure or several.

Mirrors [`JRE_VecJRE`](@ref) on the return side. A method bounded here serves a measure and a vector of them with one body, so the value-level surface needs one method per arity rather than two.

A vector is scalarised into **one** number by the `sca` keyword, which defaults to [`SumScalariser`](@ref) and is silently inert on a single measure.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`VecBaseRM`](@ref)
  - [`JRE_VecJRE`](@ref)
  - [`expected_risk`](@ref)
  - [`Scalariser`](@ref)
"""
const BaseRM_VecBaseRM = Union{<:AbstractBaseRiskMeasure, <:VecBaseRM}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether every element of `rs` shares the same polarity, and throw when they do not.

The scalarised value of a mixed vector such as `[Variance(), ExpectedReturn()]` has **no** defined orientation, so neither `all` nor `any` is correct. Both are wrong in silence, and the consumers are load-bearing: [`RankRule`](@ref) and [`QuantileRule`](@ref) take counts from each tail, and the flag decides which tail is best.

# Validation

  - Throws an `ArgumentError` when the elements disagree.

# Related

  - [`bigger_is_better`](@ref)
  - [`VecBaseRM`](@ref)
"""
function bigger_is_better(rs::VecBaseRM)::Bool
    b = bigger_is_better(first(rs))
    if !all(r -> bigger_is_better(r) == b, rs)
        got = join(["  $(nameof(typeof(r))) => $(bigger_is_better(r))" for r in rs], "\n")
        throw(ArgumentError("a vector of risk measures must agree on its polarity, and this one does not. `bigger_is_better` decides which tail of a ranking is best, so a mixed vector has no defined orientation and either answer would be wrong in silence.\nSplit the vector into one group per polarity, or rank on a single measure.\nGot\n$(got)"))
    end
    return b
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether every element of `rs` supports evaluation on a precomputed return series.

Answers by `all`, matching the `&&` the two ratio composites already use. Note the polarity against its neighbour: [`needs_previous_weights`](@ref) on the same alias answers by `any`. Both are correct — "does the vector need it" is not "can the vector do it".

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`VecBaseRM`](@ref)
  - [`expected_risk_from_returns`](@ref)
"""
function supports_precomputed_returns(rs::VecBaseRM)::Bool
    return all(supports_precomputed_returns, rs)
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for the input-shape classification of a risk measure, used by [`expected_risk`](@ref) to decide what to feed a measure's functor.

Each concrete [`AbstractBaseRiskMeasure`](@ref) declares its kind via [`risk_input_kind`](@ref). The three kinds correspond to the three functor call shapes:

  - [`NetReturnsInput`](@ref): `r(calc_net_returns(w, X, fees))`.
  - [`WeightsReturnsFeesInput`](@ref): `r(w, X, fees)`.
  - [`WeightsInput`](@ref): `r(w)`.

# Related

  - [`risk_input_kind`](@ref)
  - [`expected_risk`](@ref)
"""
abstract type RiskInputKind end
"""
$(DocStringExtensions.TYPEDEF)

Input kind for risk measures whose expected risk is computed on net returns (returns after fees). The measure's functor is called as `r(calc_net_returns(w, X, fees))`.

# Related

  - [`RiskInputKind`](@ref)
  - [`risk_input_kind`](@ref)
  - [`calc_net_returns`](@ref)
"""
struct NetReturnsInput <: RiskInputKind end
"""
$(DocStringExtensions.TYPEDEF)

Input kind for risk measures whose expected risk depends on weights, the returns matrix, and fees. The measure's functor is called as `r(w, X, fees)`.

# Related

  - [`RiskInputKind`](@ref)
  - [`risk_input_kind`](@ref)
"""
struct WeightsReturnsFeesInput <: RiskInputKind end
"""
$(DocStringExtensions.TYPEDEF)

Input kind for risk measures whose expected risk depends only on portfolio weights. The measure's functor is called as `r(w)`.

# Related

  - [`RiskInputKind`](@ref)
  - [`risk_input_kind`](@ref)
"""
struct WeightsInput <: RiskInputKind end
"""
    risk_input_kind(r::AbstractBaseRiskMeasure) -> RiskInputKind

Return the [`RiskInputKind`](@ref) of risk measure `r`, declaring what its functor consumes when [`expected_risk`](@ref) evaluates it.

There is no default: every concrete [`AbstractBaseRiskMeasure`](@ref) (other than composite measures handled by explicit `expected_risk` methods) must declare its kind beside its type definition. Returning one of [`NetReturnsInput`](@ref), [`WeightsReturnsFeesInput`](@ref), or [`WeightsInput`](@ref). An undeclared measure throws, rather than silently routing to the wrong input shape.

# Validation

  - Throws an `ArgumentError` when `r` declares no method of its own. An undeclared measure would otherwise route to the wrong input shape in silence.

# Returns

  - `RiskInputKind`: the declared input kind.

# Related

  - [`RiskInputKind`](@ref)
  - [`expected_risk`](@ref)
"""
function risk_input_kind(r::AbstractBaseRiskMeasure)
    return throw(ArgumentError("`risk_input_kind` is not defined for `$(typeof(r))`. Every concrete `AbstractBaseRiskMeasure` must declare its input kind beside its definition by adding a method returning one of `NetReturnsInput()`, `WeightsReturnsFeesInput()`, or `WeightsInput()`."))
end
"""
    range_tails(r::RiskMeasure) -> @NamedTuple{loss, gain}

Decompose a *range* risk measure into the two point measures it is the sum of.

A range measure is its base measure applied twice: once to the losses at level `alpha`, and once to the gains at the second level the range carries (`beta`, `w2`, …). The two point measures are the whole content of that statement, so a range declares them here once and both of its consumers read them from this one place:

  - [`set_range_risk_constraints!`](@ref) builds the model by calling [`set_risk_constraints!`](@ref) on `loss` with `loss = true` and on `gain` with `loss = false`, then sums the two expressions.
  - The measure's own `r(x::VecNum)` functor is the value-level twin of the same sum, `loss(x) + gain(-x)`.

The tails carry `RiskMeasureSettings(; rke = false)`: an upper bound and a risk-expression contribution belong to the range as a whole, which registers them once from the composite expression. A range that *fuses* its two tails into a shared formulation rather than duplicating one ([`OrderedWeightsArrayRange`](@ref) under [`ExactOrderedWeightsArray`](@ref), [`ValueatRiskRange`](@ref) under [`DistributionValueatRisk`](@ref)) declares no tails, because there are no two sub-models to build.

There is no default. A measure that is not a range, or one that fuses, throws rather than returning a decomposition that does not describe it.

# Arguments

  - `r`: Range risk measure.

# Validation

  - Throws an `ArgumentError` when `r` declares no method of its own. A measure that is not a range, and one that fuses its two tails into a single formulation, both take that arm.

# Returns

  - `(; loss, gain)`: The loss-tail and gain-tail point measures.

# Related

  - [`set_range_risk_constraints!`](@ref)
  - [`GenericValueatRiskRange`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
function range_tails(r::AbstractBaseRiskMeasure)
    return throw(ArgumentError("`range_tails` is not defined for `$(typeof(r))`. Only a range risk measure that is the sum of two point measures decomposes; a measure that fuses its two tails into one formulation declares none."))
end
"""
    (r::AbstractBaseRiskMeasure)(::VecNum)

Backstop for the single-argument *precomputed-returns* functor contract `r(x::VecNum)`
(ADR 0007).

This method is only ever reached by a measure that defines **no** `VecNum` functor of its
own — e.g. a composite carrying a weights-only variance term such as `VarianceSkewKurtosis`.
For such a measure the precomputed-returns form is undefined, so this throws.

It is *not* the primary safety mechanism. A [`WeightsInput`](@ref) measure's own functor
`r(w)` shares this `r(::VecNum)` signature and would otherwise silently consume a return
series *as weights*; dispatch alone cannot tell the two apart. Eligibility is therefore
decided up front by [`supports_precomputed_returns`](@ref), which the contract entry
[`expected_risk_from_returns`](@ref) consults before ever calling the functor.

# Validation

  - Throws an `ArgumentError` on every call. The method names no precondition a caller can meet: reaching it means the measure has no precomputed-returns form at all.

# Returns

  - Nothing is returned. The method always raises.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`supports_precomputed_returns`](@ref)
  - [`expected_risk_from_returns`](@ref)
"""
function (r::AbstractBaseRiskMeasure)(::VecNum)
    return throw(ArgumentError("`$(typeof(r))` has no precomputed-return-series form `r(x::VecNum)`: its risk depends on portfolio weights and/or per-asset data (e.g. a variance-carrying composite such as `VarianceSkewKurtosis`). Evaluate it through `expected_risk(r, w, X, fees)` with explicit weights instead."))
end
"""
    supports_precomputed_returns(r::AbstractBaseRiskMeasure) -> Bool
    supports_precomputed_returns(rk::RiskInputKind, r::AbstractBaseRiskMeasure) -> Bool

Whether risk measure `r` has a well-defined *precomputed-returns* form — i.e. whether its
expected risk can be evaluated on an already-reduced net-return series `x` alone, via the
functor `r(x::VecNum)` (ADR 0007).

The contract is well-defined exactly when the measure's result is a function of the series
alone:

  - [`NetReturnsInput`](@ref) measures (quantile / drawdown families): always `true` — their
    functor *is* the net-returns functor.
  - The moment family ([`LowOrderMoment`](@ref), [`HighOrderMoment`](@ref), [`Skewness`](@ref),
    [`Kurtosis`](@ref), [`MedianAbsoluteDeviation`](@ref), [`ThirdCentralMoment`](@ref)):
    `true` iff its target is weight-independent (`mu` is `nothing`, a scalar, or a centering
    function); a per-asset `mu` (`VecNum`/`VecScalar`) reduces as `dot(w, mu)` and needs the
    weights the series no longer carries, so `false`.
  - [`WeightsInput`](@ref) measures, tracking measures, and variance-carrying composites
    (`VarianceSkewKurtosis`): `false` — "risk of a bare return series" is undefined for them.

This predicate is what makes the precomputed-returns contract *safe*. Because a
`WeightsInput` measure's functor `r(w)` shares the `r(::VecNum)` signature with the contract,
dispatch alone cannot distinguish weights from returns; [`expected_risk_from_returns`](@ref)
consults this predicate and throws an explanatory error for ineligible measures rather than
silently consuming the series as weights.

# Related

  - [`expected_risk_from_returns`](@ref)
  - [`risk_input_kind`](@ref)
  - [`RiskInputKind`](@ref)
"""
function supports_precomputed_returns(r::AbstractBaseRiskMeasure)
    return supports_precomputed_returns(risk_input_kind(r), r)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`: [`NetReturnsInput`](@ref) measures always support precomputed returns —
their functor *is* the net-returns functor.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`NetReturnsInput`](@ref)
"""
supports_precomputed_returns(::NetReturnsInput, ::Any) = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: [`WeightsInput`](@ref) measures never support precomputed returns —
their functor consumes portfolio weights, not a return series.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`WeightsInput`](@ref)
"""
supports_precomputed_returns(::WeightsInput, ::Any) = false
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Erroring tripwire for [`WeightsReturnsFeesInput`](@ref) measures that have not declared
[`supports_precomputed_returns`](@ref) at their own definition site. Every such measure
must declare it there: moment measures as `supports_precomputed_returns(r::T) = weight_independent_target(r.mu)`; weights-dependent measures (tracking, variance-carrying
composites) as `supports_precomputed_returns(::T) = false`. Reaching this leaf means a
measure forgot to declare it — throws an `ArgumentError` with instructions rather than
silently mis-routing (and the completeness test in `test_09c_risk_input_kind.jl` turns that
into a CI failure).

# Validation

  - Throws an `ArgumentError` on every call. Reaching this leaf means the measure declared no `supports_precomputed_returns` method of its own, and the message states the declaration to write.

# Returns

  - Nothing is returned. The method always raises.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`WeightsReturnsFeesInput`](@ref)
  - [`weight_independent_target`](@ref)
"""
function supports_precomputed_returns(::WeightsReturnsFeesInput, r::AbstractBaseRiskMeasure)
    return throw(ArgumentError("`$(typeof(r))` is a `WeightsReturnsFeesInput` risk measure that does not declare `supports_precomputed_returns`. Declare it at the measure's definition site: a moment measure as `supports_precomputed_returns(r::$(typeof(r))) = weight_independent_target(r.mu)`; a weights-dependent measure (tracking, variance-carrying composite) as `supports_precomputed_returns(::$(typeof(r))) = false`."))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`: a `Nothing` target is trivially weight-independent and can be evaluated
on a bare return series.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
"""
weight_independent_target(::Nothing) = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true`: a scalar target does not require portfolio weights and can be evaluated
on a bare return series.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
"""
weight_independent_target(::Number) = true
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `false`: the target type requires portfolio weights (e.g. a per-asset `mu` involves
`dot(w, mu)`) and cannot be evaluated on a bare return series.

# Related

  - [`supports_precomputed_returns`](@ref)
  - [`weight_independent_target`](@ref)
"""
weight_independent_target(::Any) = false
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk measures that are not intended for use in portfolio optimisation routines.

These risk measures are typically used for analysis, reporting, or diagnostics, and are not designed to be included as objectives or constraints in optimisation problems. Subtype this when implementing a risk measure that should not be selectable by optimisation algorithms.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type NonOptimisationRiskMeasure <: AbstractBaseRiskMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk measures that are intended for use in portfolio optimisation routines.

All concrete risk measures that can be used as objectives or constraints in optimisation problems should subtype `OptimisationRiskMeasure`. This ensures compatibility with the optimisation framework and enables composability with other estimators and algorithms.

# Related

  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
  - [`NonOptimisationRiskMeasure`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
"""
abstract type OptimisationRiskMeasure <: AbstractBaseRiskMeasure end
"""
    const VecOptRM = AbstractVector{<:OptimisationRiskMeasure}

Alias for an abstract vector of [`OptimisationRiskMeasure`](@ref) elements.

# Related

  - [`OptimisationRiskMeasure`](@ref)
  - [`OptRM_VecOptRM`](@ref)
  - [`VecRM`](@ref)
"""
const VecOptRM = AbstractVector{<:OptimisationRiskMeasure}
"""
    const OptRM_VecOptRM = Union{<:OptimisationRiskMeasure, <:VecOptRM}

Union type accepting a single [`OptimisationRiskMeasure`](@ref) or a vector of them.

# Related

  - [`OptimisationRiskMeasure`](@ref)
  - [`VecOptRM`](@ref)
"""
const OptRM_VecOptRM = Union{<:OptimisationRiskMeasure, <:VecOptRM}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for standard risk measures used in portfolio optimisation.

Subtype `RiskMeasure` to implement concrete risk measures that quantify portfolio risk and can be used as objectives or constraints in optimisation problems. This type ensures compatibility with the optimisation framework and enables composability with other estimators and algorithms.

# Related

  - [`OptimisationRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type RiskMeasure <: OptimisationRiskMeasure end
"""
    const VecRM = AbstractVector{<:RiskMeasure}

Alias for an abstract vector of [`RiskMeasure`](@ref) elements.

# Related

  - [`RiskMeasure`](@ref)
  - [`RM_VecRM`](@ref)
  - [`VecOptRM`](@ref)
"""
const VecRM = AbstractVector{<:RiskMeasure}
"""
    const RM_VecRM = Union{<:RiskMeasure, <:VecRM}

Union type accepting a single [`RiskMeasure`](@ref) or a vector of them.

# Related

  - [`RiskMeasure`](@ref)
  - [`VecRM`](@ref)
"""
const RM_VecRM = Union{<:RiskMeasure, <:VecRM}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for hierarchical risk measures used in portfolio optimisation.

Subtype `HierarchicalRiskMeasure` to implement risk measures that operate on hierarchical or clustered portfolio structures. These measures are designed for use as objectives or constraints in optimisation problems that leverage asset clustering, hierarchical risk parity, or similar techniques.

# Related

  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
"""
abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all risk measure settings.

A settings object carries the three things a risk measure needs that are not part of the risk itself: the combination weight `scale`, whether the measure contributes to the model's risk expression (`rke`), and the bound the optimiser enforces on the measure's own expression. All concrete risk measure settings types should subtype `AbstractRiskMeasureSettings`.

# Interfaces

In order to implement a new concrete settings type, subtype `AbstractRiskMeasureSettings` and give it these fields:

## Required fields

  - `scale::Number`: Combination weight, read by [`set_risk_expression!`](@ref) and by the value-level readers.
  - `rke::Bool`: Whether the measure contributes to the model's aggregate risk expression.

## Optional field

  - A bound field. [`RiskMeasureSettings`](@ref) names it `ub` and bounds the expression from above; [`MaxRiskMeasureSettings`](@ref) names it `lb` and bounds it from below. [`HierarchicalRiskMeasureSettings`](@ref) has none, because a clustering optimiser builds no model to bound.

# Related

  - [`JuMPRiskMeasureSettings`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
"""
abstract type AbstractRiskMeasureSettings <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk measure settings used in JuMP-based optimisation routines.

All concrete settings types compatible with JuMP optimisation (e.g. [`RiskMeasureSettings`](@ref), [`MaxRiskMeasureSettings`](@ref)) should subtype `JuMPRiskMeasureSettings`.

# Related

  - [`AbstractRiskMeasureSettings`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
"""
abstract type JuMPRiskMeasureSettings <: AbstractRiskMeasureSettings end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for bound-transformation strategies applied to efficient frontier bounds.

A bound and the risk expression it bounds are not always in the same units. A concrete subtype names the conversion between them, and it is used twice: once on a stated bound value, and once on each end of a [`Frontier`](@ref) sweep span. Because the span is transformed before it is divided, the sweep points are evenly spaced in the units of the **expression**, not in the units of the measure.

# Interfaces

In order to implement a new bound transformation, subtype `FrontierBoundEstimator` and implement:

## Required method name

  - `variance_risk_bounds_val(bound::MyBound, ub::Num_VecNum)`: Convert a stated bound into the units of the risk expression.

### Arguments

  - `bound`: The bound-transformation instance.
  - `ub`: The bound value the caller stated.

### Returns

  - The converted bound, in the units of the risk expression.

The [`Frontier`](@ref) arm of `variance_risk_bounds_val` is generic and needs no method. The sweep applies the same transformation to both ends of its span in [`rebuild_risk_frontier`](@ref), which reads `bound` off the `Frontier` and branches on it there.

# Related

  - [`LinearBound`](@ref)
  - [`SquareRootBound`](@ref)
  - [`SquaredBound`](@ref)
  - [`variance_risk_bounds_val`](@ref)
  - [`Frontier`](@ref)
"""
abstract type FrontierBoundEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Applies a square-root transformation to bound values before enforcing them.

The conversion is `sqrt(ub)`. The caller writes the bound against the **measure**, and the model enforces it against the square root of that quantity, so a bound on a variance becomes a bound on a standard deviation. Used when the risk expression is in standard-deviation units but the user-supplied bound is in variance units (e.g. kurtosis and negative-skewness SOC formulations). This is also the transformation the default [`Variance`](@ref) formulation takes, so a `Variance` frontier is evenly spaced in **standard-deviation** units: on a 250x5 sample a five-point sweep gave standard deviations `0.004539514, 0.005969048, 0.007398587, 0.008828127, 0.010257666`, evenly spaced to `4.1e-9`, whose squares are the variances `2.0607e-5, 3.5630e-5, 5.4739e-5, 7.7936e-5, 1.0522e-4`, which are **not** evenly spaced.

# Related

  - [`FrontierBoundEstimator`](@ref)
  - [`LinearBound`](@ref)
  - [`SquaredBound`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
struct SquareRootBound <: FrontierBoundEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Passes bound values through unchanged (identity transformation).

The conversion is `ub` itself. The caller writes the bound against the **measure**, and the model enforces it against that same quantity. Used when the risk expression and the user-supplied bound are already in the same units (e.g. SDP variance formulation where both sides are in variance units). This is the default of [`Frontier`](@ref), so a sweep that names no `bound` is evenly spaced in the units of the measure.

# Related

  - [`FrontierBoundEstimator`](@ref)
  - [`SquareRootBound`](@ref)
  - [`SquaredBound`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
struct LinearBound <: FrontierBoundEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Applies a squaring transformation to bound values before enforcing them.

The conversion is `ub^2`. The caller writes the bound against the **measure**, and the model enforces it against the square of that quantity, so a bound on a standard deviation becomes a bound on a variance. Used when the risk expression is in squared units but the user-supplied bound is in linear units (e.g. kurtosis SDP formulation).

# Related

  - [`FrontierBoundEstimator`](@ref)
  - [`LinearBound`](@ref)
  - [`SquareRootBound`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
struct SquaredBound <: FrontierBoundEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Sweeps the efficient frontier by solving the model once at each of `N` evenly spaced bound values.

Stand this in a risk measure's bound slot instead of a number. The optimisation then reads the span off two corner portfolios, divides it into `N` levels, and returns one portfolio per level. The corners are a minimum-risk solve and a maximum-return solve, so the span covers the whole attainable range of the bound.

The levels are evenly spaced in the units of the **risk expression** the bound is applied to, which `bound` names, and not in the units of the measure. See [`SquareRootBound`](@ref) for the case where the two differ.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    Frontier(;
        N::Integer = 20,
        bound::FrontierBoundEstimator = LinearBound()
    ) -> Frontier

Creates a `Frontier` with `N` points, a scaling factor of `1`, and the specified `bound` strategy. Used to set appropriate frontier bounds in [`variance_risk_bounds_val`](@ref) and [`second_moment_bound_val`](@ref).

    PortfolioOptimisers._Frontier(;
        N::Integer = 20,
        factor::Number,
        bound::FrontierBoundEstimator
    ) -> Frontier

Internal constructor. Keywords correspond to the struct's fields.

## Validation

  - `N > 0` and `N <= RESOURCE_LIMITS[].max_frontier` (each sweep point runs a full solve; see [`RESOURCE_LIMITS`](@ref)). This is the cheap early check on **one** bound. The sweep across several swept bounds is their product, capped separately at Model Assembly by [`assert_frontier_sweep_cap`](@ref).
  - `isfinite(factor)` and `factor > 0`.

# Examples

```jldoctest
julia> Frontier(; N = 15)
Frontier
       N ┼ Int64: 15
  factor ┼ Int64: 1
   bound ┴ LinearBound()
```

# Related

  - [`FrontierBoundEstimator`](@ref)
  - [`LinearBound`](@ref)
  - [`SquareRootBound`](@ref)
  - [`SquaredBound`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`rebuild_risk_frontier`](@ref)

# References

  - $(ref_dict[:cajas2025]) Sections 8.2.1 and 8.2.2, Equations 8.7 and 8.10.
"""
@concrete struct Frontier <: AbstractAlgorithm
    """
    $(field_dict[:N_fr])
    """
    N
    """
    $(field_dict[:factor_fr])
    """
    factor
    """
    $(field_dict[:bound_fr])
    """
    bound
    function Frontier(N::Integer, factor::Number = 1,
                      bound::FrontierBoundEstimator = LinearBound())::Frontier
        @argcheck(N > zero(N), DomainError(N, "N must be > 0"))
        assert_resource_cap(N, RESOURCE_LIMITS[].max_frontier, :N, :max_frontier)
        @argcheck(isfinite(factor), IsNonFiniteError("factor must be finite, got $factor"))
        @argcheck(factor > zero(factor), DomainError(factor, "factor must be positive"))
        return new{typeof(N), typeof(factor), typeof(bound)}(N, factor, bound)
    end
end
function Frontier(; N::Integer = 20,
                  bound::FrontierBoundEstimator = LinearBound())::Frontier
    return Frontier(N, 1, bound)
end
"""
    _Frontier(; N = 20, factor, bound)

Build a [`Frontier`](@ref) whose `factor` and `bound` a formulation chose, rather than the caller.

The keyword constructor `Frontier(; N, bound)` pins `factor` to `1`, because a caller states a bound in the units of the measure and knows nothing of the model's internal scaling. A formulation does know: it reads the caller's `Frontier`, keeps its `N`, and rebuilds it with the conversion its own risk expression needs. That is what [`variance_risk_bounds_val`](@ref) and [`second_moment_bound_val`](@ref) do.

This builds a `Frontier`, not the sweep values. The values are built later, in [`rebuild_risk_frontier`](@ref), once the two corner portfolios are known: `bound` transforms each end of the span, `factor` multiplies both, and `range` divides the result into `N` levels.

# Arguments

  - `N`: Number of sweep points (default 20).
  - `factor`: Multiplier applied to both ends of the span, after `bound` has transformed them.
  - `bound`: [`FrontierBoundEstimator`](@ref) converting the span into the units of the risk expression.

# Returns

  - `Frontier`: The rebuilt sweep specification.

# Related

  - [`Frontier`](@ref)
  - [`variance_risk_bounds_val`](@ref)
  - [`second_moment_bound_val`](@ref)
  - [`rebuild_risk_frontier`](@ref)
  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function _Frontier(; N::Integer = 20, factor::Number, bound::FrontierBoundEstimator)
    return Frontier(N, factor, bound)
end
"""
    const RkRtBounds = Union{<:Num_VecNum, <:Frontier}

Union type for risk-measure upper bound specifications.

Accepts either a scalar/vector numeric bound or a [`Frontier`](@ref) sweep configuration. Used in [`RiskMeasureSettings`](@ref) to set the upper bound field.

# Related

  - [`Frontier`](@ref)
  - [`Num_VecNum`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
const RkRtBounds = Union{<:Num_VecNum, <:Frontier}
"""
    const Front_NumVec = Union{<:VecNum, <:Frontier}

Union type for frontier or numeric-vector specifications used internally for risk bounds.

# Related

  - [`Frontier`](@ref)
  - [`VecNum`](@ref)
  - [`RkRtBounds`](@ref)
"""
const Front_NumVec = Union{<:VecNum, <:Frontier}
"""
$(DocStringExtensions.TYPEDEF)

Weights a risk measure inside an aggregate, and bounds its risk expression from above.

This is the settings type of every measure the optimiser minimises. `ub` is the maximum level of risk the solution may reach, the ``\\bar{\\phi}`` of Equation 8.10 of the reference below; sweeping it is how the book builds an efficient frontier, which is what a [`Frontier`](@ref) in that slot does. `scale` weights the measure against its siblings when several are given, and `rke` decides whether it reaches the aggregate at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RiskMeasureSettings(;
        scale::Number = 1.0,
        ub::Option{<:RkRtBounds} = nothing,
        rke::Bool = true,
    ) -> RiskMeasureSettings

Creates a `RiskMeasureSettings` instance with the specified scale, upper bound, and risk evaluation flag.

## Validation

  - `isfinite(scale)`.
  - `ub` is validated with [`assert_nonempty_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> RiskMeasureSettings()
RiskMeasureSettings
  scale ┼ Float64: 1.0
     ub ┼ nothing
    rke ┴ Bool: true
```

# Related

  - [`JuMPRiskMeasureSettings`](@ref)
  - [`RiskMeasure`](@ref)
  - [`Frontier`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`MaxRiskMeasureSettings`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 8.2.2, Equation 8.10.
"""
@concrete struct RiskMeasureSettings <: JuMPRiskMeasureSettings
    """
    $(field_dict[:scale_rm])
    """
    scale
    """
    $(field_dict[:ub_rms])
    """
    ub
    """
    $(field_dict[:rke])
    """
    rke
    function RiskMeasureSettings(scale::Number, ub::Option{<:RkRtBounds},
                                 rke::Bool)::RiskMeasureSettings
        assert_nonempty_nonneg_finite_val(ub, :ub)
        @argcheck(isfinite(scale), IsNonFiniteError("scale must be finite, got $scale"))
        return new{typeof(scale), typeof(ub), typeof(rke)}(scale, ub, rke)
    end
end
function RiskMeasureSettings(; scale::Number = 1.0, ub::Option{<:RkRtBounds} = nothing,
                             rke::Bool = true)::RiskMeasureSettings
    return RiskMeasureSettings(scale, ub, rke)
end
"""
$(DocStringExtensions.TYPEDEF)

Weights a hierarchical risk measure inside an aggregate, and carries no bound.

A clustering optimiser computes each cluster's risk as a number rather than building one model over the whole universe, so there is no risk expression for a bound to constrain and no `ub` field to hold one. `scale` is the only setting a [`HierarchicalRiskMeasure`](@ref) has, and it means what it means everywhere else: the measure's weight against its siblings when several are given.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskMeasureSettings(;
        scale::Number = 1.0
    ) -> HierarchicalRiskMeasureSettings

Creates a `HierarchicalRiskMeasureSettings` instance with the specified scaling factor.

## Validation

  - `scale` must be finite.

# Examples

```jldoctest
julia> HierarchicalRiskMeasureSettings()
HierarchicalRiskMeasureSettings
  scale ┴ Float64: 1.0
```

# Related

  - [`AbstractRiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
@concrete struct HierarchicalRiskMeasureSettings <: AbstractRiskMeasureSettings
    """
    $(field_dict[:scale_rm])
    """
    scale
    function HierarchicalRiskMeasureSettings(scale::Number)::HierarchicalRiskMeasureSettings
        @argcheck(isfinite(scale), IsNonFiniteError("scale must be finite, got $scale"))
        return new{typeof(scale)}(scale)
    end
end
function HierarchicalRiskMeasureSettings(;
                                         scale::Number = 1.0)::HierarchicalRiskMeasureSettings
    return HierarchicalRiskMeasureSettings(scale)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the risk measure `rs` unchanged.

Identity pass-through used when a risk measure is provided in a context that calls [`factory`](@ref).

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`factory`](@ref)
"""
function factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
    return rs
end
"""
    port_opt_view(rs, i, X)

Get a view or subset of a risk measure for asset cluster index `i`.

Returns the risk measure sliced for the given cluster or asset index. Used internally in hierarchical optimisation to apply risk measures to each cluster.

A vector of risk measures ([`VecBaseRM`](@ref)) is handled by the generic vector methods of [`factory`](@ref) and [`port_opt_view`](@ref), which rebuild and view each measure in turn.

# Arguments

  - `rs`: Risk measure.
  - `i`: Cluster or asset index.
  - `X`: Data matrix (used for dimension-aware slicing).

# Returns

  - Sliced risk measure or the original if no slicing is needed.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`VecBaseRM`](@ref)
"""
function port_opt_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any,
                       args...)::AbstractBaseRiskMeasure
    return rs
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies used to combine multiple risk measures into a single scalar value for optimisation.

Subtype `Scalariser` to implement different methods for aggregating risk measures. These strategies are used in portfolio optimisation routines that require a single risk value from multiple risk measures.

## Two consumers, and only one of them is restricted

A scalariser is read at two levels, and they are not equally permissive.

  - **The model level.** A `JuMP` optimiser builds the aggregation into the model as an expression, so the strategy must have a convex form. This is what [`NonHierarchicalScalariser`](@ref) and [`HierarchicalScalariser`](@ref) separate: the two subtypes name the **consumers** that can build a given strategy, not a property of the resulting number. `JuMPOptimiser.sca` is bounded `NonHierarchicalScalariser` and refuses the hierarchical half; the clustering optimisers compute each cluster's risk separately and accept either.
  - **The value level.** [`expected_risk`](@ref) and the readers around it evaluate the measures first and combine the resulting **numbers**, so nothing convex is being built and no strategy can be refused on those grounds. Every `sca` keyword and every `sca` result field at this level is bounded [`Scalariser`](@ref), and **all four scalarisers are admitted**, [`MinScalariser`](@ref) included.

So a `HierarchicalScalariser` on a value-level call is not a misuse, and the word "hierarchical" in the subtype name describes which estimators accept it rather than where the number is meaningful.

## The weights are the measures' own `scale`

Every strategy below is a **weighted** aggregation, and it reads its weights from one place: the `scale` field of each measure's settings. Both levels apply it before the strategy runs — the model in [`set_risk_expression!`](@ref), which pushes `scale * r_expr`, and the value level in [`expected_risk`](@ref), which multiplies each element by `r.settings.scale`. Stating it once here is why no strategy below carries a weight vector of its own.

## Two of the three model forms are exact only while the objective pushes

[`SumScalariser`](@ref) builds an expression, so the model agrees with the value level whatever the objective is. The other two build a **variable** bounded below by the aggregation, and a minimising objective is what pulls it down onto the aggregation. Under [`MaximumReturn`](@ref) nothing pulls, so `model[:risk]` reports an upper bound: on a 250x5 sample [`MaxScalariser`](@ref) reported `0.0782482` against a true `0.0380465`, and [`LogSumExpScalariser`](@ref) at `gamma = 100` reported `0.0612520` against a true `0.0440182`.

The bound the model **enforces** is unaffected. An `ub` on the aggregate constrains the variable, and the variable stands above the aggregation, so the aggregation satisfies the bound too. It is the reported figure that stands above it, and reading the aggregate back from [`expected_risk`](@ref) gives the exact one.

# Interfaces

In order to implement a new scalarisation strategy, subtype [`NonHierarchicalScalariser`](@ref) when the aggregation has a convex `JuMP` form and [`HierarchicalScalariser`](@ref) when it does not, then implement:

## Required method name

  - `scalarise(f, sca::MyScalariser, itr; by = nothing)`: Combine the per-measure values `f` returns into one.

### Arguments

  - `f`: Per-element evaluation closure.
  - `sca`: The strategy instance.
  - `itr`: Iterable of risk measures.
  - `by`: Optional selection key, for a strategy that picks one element rather than combining all of them.

### Returns

  - The combined value, with the same shape as the values `f` returns.

## Required for a `NonHierarchicalScalariser`

  - `scalarise_risk_expression!(model::JuMP.Model, sca::MyScalariser)`: Collapse the model's `risk_vec` array into the single `risk` expression. A [`HierarchicalScalariser`](@ref) declares no such method, which is what the split means.

# Related

  - [`NonHierarchicalScalariser`](@ref)
  - [`HierarchicalScalariser`](@ref)
  - [`scalarise`](@ref)
  - [`scalarise_risk_expression!`](@ref)

# References

  - $(ref_dict[:boydvandenberghe2004]) Vector optimisation and scalarisation.
  - $(ref_dict[:diamondboyd2016]) The `cvxpy.transforms.scalarize` module, which this family ports.
"""
abstract type Scalariser <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies that combine multiple risk measures into a single scalar value, and that **every** optimisation estimator can build.

Subtype `NonHierarchicalScalariser` when the aggregation has a convex `JuMP` form, so a `JuMP` optimiser can build it into the model as well as a clustering optimiser can compute it per cluster. `JuMPOptimiser.sca` is bounded here.

The split names a **consumer**, not a property of the number. It does not restrict the value-level readers, which admit all four scalarisers — see [`Scalariser`](@ref).

# Related

  - [`SumScalariser`](@ref)
  - [`MaxScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`Scalariser`](@ref)
  - [`HierarchicalScalariser`](@ref)
"""
abstract type NonHierarchicalScalariser <: Scalariser end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies that combine multiple risk measures into a single scalar value, and that **only the clustering optimisation estimators** can build.

Subtype `HierarchicalScalariser` when the aggregation has no convex `JuMP` form. `JuMPOptimiser.sca` is bounded [`NonHierarchicalScalariser`](@ref) and refuses this half; the clustering optimisers compute each cluster's risk separately and accept it.

The split names a **consumer**, not a property of the number. At the value level the measures are evaluated first and the strategy combines plain numbers, so nothing is being built and all four scalarisers are admitted — see [`Scalariser`](@ref).

# Related

  - [`MinScalariser`](@ref)
  - [`Scalariser`](@ref)
  - [`NonHierarchicalScalariser`](@ref)
"""
abstract type HierarchicalScalariser <: Scalariser end
"""
$(DocStringExtensions.TYPEDEF)

Adds the scaled risk measures together.

This is the default everywhere a `sca` keyword appears, and the only strategy whose model form is an expression rather than a variable with constraints: [`scalarise_risk_expression!`](@ref) sums `risk_vec` into one affine or quadratic expression, so it is also the only one that takes a quadratic risk expression without a reformulation. On a 250x5 sample the model and [`expected_risk`](@ref) agreed to `3.6e-9` relative.

# Mathematical definition

```math
\\begin{align}
\\phi &= \\sum_{i=1}^{N} w_i \\cdot r_i \\,.
\\end{align}
```

Where:

  - ``N``: Number of risk measures.
  - ``i``: Subscript denoting the `i`-th risk measure.
  - ``r_i``: `i`-th risk measure value.
  - ``w_i``: Weight of the `i`-th risk measure.

# Related

  - [`Scalariser`](@ref)
  - [`MaxScalariser`](@ref)
  - [`MinScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`scalarise_risk_expression!`](@ref)

# References

  - $(ref_dict[:boydvandenberghe2004]) Vector optimisation and scalarisation.
  - $(ref_dict[:diamondboyd2016]) `cvxpy.transforms.scalarize.weighted_sum`.
"""
struct SumScalariser <: NonHierarchicalScalariser end
"""
$(DocStringExtensions.TYPEDEF)

Reports the largest of the scaled risk measures, so the aggregate is the worst of them.

The model form is a free variable held above every entry of `risk_vec`, which the objective then pushes down onto the largest. Under a minimum-risk objective on a 250x5 sample the model and [`expected_risk`](@ref) agreed to `2.1e-7` relative; under a maximum-return objective on the same sample the model reported `0.0782482` against a true `0.0380465`, which is the shared caveat in [`Scalariser`](@ref). In clustering optimisations each cluster's risk is computed separately, so there is no coherence in which measure wins between clusters.

# Mathematical definition

```math
\\begin{align}
\\phi &= \\underset{i \\in (1,\\,N)}{\\max} \\left(w_i \\cdot r_i \\right)\\,.
\\end{align}
```

Where:

  - ``N``: Number of risk measures.
  - ``i``: Subscript denoting the `i`-th risk measure.
  - ``r_i``: `i`-th risk measure value.
  - ``w_i``: Weight of the `i`-th risk measure.

# JuMP formulation

## Relaxation

$(val_dict[:relax])

  - The bounded quantity is `risk`, the free variable [`scalarise_risk_expression!`](@ref) creates for this strategy. The row `risk_ms` holds it at or **above** every entry of `risk_vec`, so `model[:risk]` stands above the maximum rather than on it.
  - The bound is tight while the objective pulls `risk` down, which a minimum-risk objective does. Under [`MaximumReturn`](@ref) nothing pulls: on a 250x5 sample the model reported `0.0782482` against a true `0.0380465`.
  - An `ub` on the aggregate is unaffected. It constrains `risk`, and the aggregation stands below `risk`, so the aggregation meets the bound too. Read the exact figure back with [`expected_risk`](@ref).

# Related

  - [`Scalariser`](@ref)
  - [`SumScalariser`](@ref)
  - [`MinScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`scalarise_risk_expression!`](@ref)

# References

  - $(ref_dict[:boydvandenberghe2004]) Vector optimisation and scalarisation.
  - $(ref_dict[:diamondboyd2016]) `cvxpy.transforms.scalarize.max`.
"""
struct MaxScalariser <: NonHierarchicalScalariser end
"""
$(DocStringExtensions.TYPEDEF)

Reports the smallest of the scaled risk measures, so the aggregate is the mildest of them.

It is this library's own extension: the source family carries a weighted sum, a maximum and a log-sum-exp, and no minimum. In clustering optimisations each cluster's risk is computed separately, so there is no coherence in which measure wins between clusters.

It is the one [`HierarchicalScalariser`](@ref), because minimising a minimum is not convex and a `JuMP` optimiser cannot build it. That bound is the **model's**: at the value level the measures are already numbers, so `MinScalariser` is admitted by every `sca` keyword and every `sca` result field there, on equal footing with the other three.

# Mathematical definition

```math
\\begin{align}
\\phi &= \\underset{i \\in (1,\\,N)}{\\min} \\left( w_i \\cdot r_i \\right)\\,.
\\end{align}
```

Where:

  - ``N``: Number of risk measures.
  - ``i``: Subscript denoting the `i`-th risk measure.
  - ``r_i``: `i`-th risk measure value.
  - ``w_i``: Weight of the `i`-th risk measure.

# Related

  - [`Scalariser`](@ref)
  - [`SumScalariser`](@ref)
  - [`MaxScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
"""
struct MinScalariser <: HierarchicalScalariser end
"""
$(DocStringExtensions.TYPEDEF)

Smooths the maximum of the scaled risk measures, so every measure keeps a share of the aggregate.

# Mathematical definition

```math
\\begin{align}
\\phi &= \\frac{1}{\\gamma} \\log \\left( \\sum_{i=1}^{N} \\exp \\left[ \\gamma \\cdot w_i \\cdot r_i \\right] \\right)\\,.
\\end{align}
```

Where:

  - ``N``: Number of risk measures.
  - ``i``: Subscript denoting the `i`-th risk measure.
  - ``r_i``: `i`-th risk measure value.
  - ``w_i``: Weight of the `i`-th risk measure.
  - ``\\gamma``: Positive parameter setting how close the aggregate runs to the maximum.

## What `gamma` does

The aggregate is never below the maximum and never more than ``\\log N / \\gamma`` above it:

```math
\\begin{align}
\\underset{i \\in (1,\\,N)}{\\max} \\left(w_i \\cdot r_i\\right) \\leq \\phi &\\leq \\underset{i \\in (1,\\,N)}{\\max} \\left(w_i \\cdot r_i\\right) + \\frac{\\log N}{\\gamma}\\,.
\\end{align}
```

So a large ``\\gamma`` gives the maximum. On the scaled values `[0.1, 0.2, 0.05]`, ``\\gamma = 100`` returns `0.20000046` against a maximum of `0.2`, inside the bound ``\\log 3 / 100 = 0.010986``. In a model at ``\\gamma = 100`` under a minimum-risk objective the model and [`expected_risk`](@ref) agreed to `4.1e-9` relative, and under a maximum-return objective the model reported `0.0612520` against a true `0.0440182` — the shared caveat in [`Scalariser`](@ref).

!!! warning

    A small ``\\gamma`` does **not** give the weighted sum. The same three values return `1098.73` at ``\\gamma = 0.001``, against a weighted sum of `0.35`. The bound above shows why: the aggregate never falls below the maximum, and ``\\log N / \\gamma`` diverges. What survives is the shape — subtracting that divergent term leaves `0.11666861`, the weighted **mean** `0.11666667`. An additive constant does not move a minimiser, so the *portfolio* a small ``\\gamma`` selects tends to the weighted sum's, while the *number* reported does not. The model degenerates first: with the risks near `1e-2` and the objective near ``\\log 2``, a two-measure model at ``\\gamma = 1`` already failed to solve.

!!! warning

    The model form is an exponential cone, which cannot hold a quadratic risk expression. Both [`Variance`](@ref) formulations produce one, so a `Variance` under this scalariser aborts the solve with `MOI.UnsupportedConstraint{MOI.ScalarQuadraticFunction{Float64}, MOI.GreaterThan{Float64}}`. Use [`StandardDeviation`](@ref) instead, or [`SumScalariser`](@ref), which sums a quadratic expression directly. Nothing refuses the combination up front.

In clustering optimisations each cluster's risk is computed separately, so there is no coherence between clusters, and the value level has no cone to build: it evaluates the measures first and reduces the numbers, where a large ``\\gamma`` is safe because `LogExpFunctions.logsumexp` shifts by the maximum before it exponentiates.

# JuMP formulation

## Relaxation

$(val_dict[:relax])

  - The bounded quantity is `risk`, the free variable [`scalarise_risk_expression!`](@ref) creates for this strategy. The rows `u_risk_lse` and `risk_lse` state the exponential-cone form of the log-sum-exp, which holds `model[:risk]` at or **above** the smoothed maximum.
  - The bound is tight while the objective pulls `risk` down, which a minimum-risk objective does. Under [`MaximumReturn`](@ref) nothing pulls: on a 250x5 sample at ``\\gamma = 100`` the model reported `0.0612520` against a true `0.0440182`.
  - An `ub` on the aggregate is unaffected, on the terms [`MaxScalariser`](@ref) states. Read the exact figure back with [`expected_risk`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    LogSumExpScalariser(;
        gamma::Number = 1.0,
    ) -> LogSumExpScalariser

Keywords correspond to the struct's fields.

## Validation

  - `gamma > 0`.

# Examples

```jldoctest
julia> LogSumExpScalariser()
LogSumExpScalariser
  gamma ┴ Float64: 1.0
```

# Related

  - [`Scalariser`](@ref)
  - [`SumScalariser`](@ref)
  - [`MaxScalariser`](@ref)
  - [`MinScalariser`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
  - [`scalarise_risk_expression!`](@ref)

# References

  - $(ref_dict[:boydvandenberghe2004]) The log-sum-exp function and its bound on the maximum.
  - $(ref_dict[:diamondboyd2016]) `cvxpy.transforms.scalarize.log_sum_exp`.
"""
@concrete struct LogSumExpScalariser <: NonHierarchicalScalariser
    """
    $(field_dict[:gamma])
    """
    gamma
    function LogSumExpScalariser(gamma::Number)
        @argcheck(gamma > zero(gamma), DomainError(gamma, "gamma must be positive"))
        return new{typeof(gamma)}(gamma)
    end
end
function LogSumExpScalariser(; gamma::Number = 1.0)
    return LogSumExpScalariser(gamma)
end
"""
    scalarise_combine(op, a, b)

Combine two scalarised risk values slot-wise.

Numbers combine directly via `op`, arrays elementwise, and tuples recursively per slot. Used by [`scalarise`](@ref) to accumulate values that may be scalars, vectors, or tuples mixing both.

# Related

  - [`scalarise`](@ref)
  - [`scalarise_map`](@ref)
"""
scalarise_combine(op, a::Number, b::Number) = op(a, b)
scalarise_combine(op, a::AbstractArray, b::AbstractArray) = op.(a, b)
function scalarise_combine(op, a::Tuple, b::Tuple)
    return map((x, y) -> scalarise_combine(op, x, y), a, b)
end
"""
    scalarise_map(op, x)

Apply `op` slot-wise to a scalarised risk value.

Numbers are transformed directly, arrays elementwise, and tuples recursively per slot.

# Related

  - [`scalarise`](@ref)
  - [`scalarise_combine`](@ref)
"""
scalarise_map(op, x::Number) = op(x)
scalarise_map(op, x::AbstractArray) = op.(x)
scalarise_map(op, x::Tuple) = map(y -> scalarise_map(op, y), x)
"""
    scalarise_logsumexp(vs)

Slot-wise log-sum-exp across a vector of scalarised risk values.

For a vector of numbers this is `LogExpFunctions.logsumexp`; for a vector of same-shaped arrays it is applied elementwise across the vector; for a vector of same-shaped tuples it recurses per slot.

# Related

  - [`scalarise`](@ref)
  - [`LogSumExpScalariser`](@ref)
"""
scalarise_logsumexp(vs::AbstractVector{<:Number}) = LogExpFunctions.logsumexp(vs)
function scalarise_logsumexp(vs::AbstractVector{<:AbstractArray})
    return map(i -> LogExpFunctions.logsumexp([v[i] for v in vs]), eachindex(first(vs)))
end
function scalarise_logsumexp(vs::AbstractVector{<:Tuple})
    return ntuple(k -> scalarise_logsumexp([v[k] for v in vs]), Val(length(first(vs))))
end
"""
    scalarise(f, sca::Scalariser, itr; by = nothing)

Reduce per-risk-measure values into a single scalarised value.

Applies `f` to every element of `itr` (typically a vector of risk measures, or `pairs` thereof) and combines the results according to the scalariser. `f` must return a `Number`, an `AbstractArray`, or a `Tuple` of these, and must return freshly allocated values (no views into buffers reused across iterations), since results may be retained across iterations.

The combining rules are:

  - [`SumScalariser`](@ref): slot-wise sum of all values. `f` is expected to include the measure's `settings.scale` weight in its result.

  - [`MaxScalariser`](@ref)/[`MinScalariser`](@ref):

      + `by === nothing`: slot-wise (elementwise) maximum/minimum across values.
      + `by` given: winner-take-all — returns the single `f` result whose `by(result)` is largest/smallest (ties keep the earliest).

  - [`LogSumExpScalariser`](@ref): slot-wise `logsumexp` of `gamma`-scaled values, divided by `gamma`.

# Algorithm

 1. Start the accumulator `acc` at `nothing`, and the winning key `k` at `nothing` beside it. [`LogSumExpScalariser`](@ref) starts neither, and takes steps 5 and 6 instead.
 2. Apply `f` to each element of `itr` in turn, giving `v`.
 3. Fold `v` into `acc` through the branch that `sca` selects. [`SumScalariser`](@ref) takes `scalarise_combine(+, acc, v)`. [`MaxScalariser`](@ref) and [`MinScalariser`](@ref) take `scalarise_combine(max, acc, v)` and `scalarise_combine(min, acc, v)` while `by` is `nothing`.
 4. Given a `by`, read the key `ki = by(v)` in place of step 3, and take `v` and `ki` as the new `acc` and `k` when `ki` beats the `k` already held. A tie keeps the earlier element.
 5. Under [`LogSumExpScalariser`](@ref), apply `f` to every element of `itr`, scale each result by `sca.gamma` with [`scalarise_map`](@ref), and collect them into `vs`.
 6. Reduce `vs` slot-wise with [`scalarise_logsumexp`](@ref), and divide the reduction by `sca.gamma` with [`scalarise_map`](@ref).
 7. Return `acc`, or the value step 6 built.

# Arguments

  - `f`: Per-element evaluation closure, applied to each element of `itr`.
  - `sca`: Scalarisation strategy.
  - `itr`: Iterable of risk measures (or `pairs` of them, when `f` needs the index).
  - `by`: Optional selection key for `MaxScalariser`/`MinScalariser`; ignored by the other scalarisers.

# Returns

  - The scalarised value, with the same shape as the values returned by `f`.

# Related

  - [`Scalariser`](@ref)
  - [`scalarise_combine`](@ref)
  - [`scalarise_map`](@ref)
  - [`scalarise_logsumexp`](@ref)
"""
function scalarise(f, ::SumScalariser, itr; by = nothing)
    acc = nothing
    for el in itr
        v = f(el)
        acc = acc === nothing ? v : scalarise_combine(+, acc, v)
    end
    return acc
end
function scalarise(f, ::MaxScalariser, itr; by = nothing)
    acc, k = nothing, nothing
    for el in itr
        v = f(el)
        if by === nothing
            acc = acc === nothing ? v : scalarise_combine(max, acc, v)
        else
            ki = by(v)
            if k === nothing || ki > k
                acc, k = v, ki
            end
        end
    end
    return acc
end
function scalarise(f, ::MinScalariser, itr; by = nothing)
    acc, k = nothing, nothing
    for el in itr
        v = f(el)
        if by === nothing
            acc = acc === nothing ? v : scalarise_combine(min, acc, v)
        else
            ki = by(v)
            if k === nothing || ki < k
                acc, k = v, ki
            end
        end
    end
    return acc
end
function scalarise(f, sca::LogSumExpScalariser, itr; by = nothing)
    vs = [scalarise_map(x -> sca.gamma * x, f(el)) for el in itr]
    return scalarise_map(x -> x / sca.gamma, scalarise_logsumexp(vs))
end
"""
    nothing_scalar_array_selector(risk_variable::Nothing, prior_variable::Nothing)
    nothing_scalar_array_selector(risk_variable::Num_ArrNum_VecScalar_DynWeights, ::Any)
    nothing_scalar_array_selector(risk_variable::Nothing, prior_variable::Num_ArrNum_VecScalar_DynWeights)

Function for selecting a non-nothing value when provided by a risk measure, or fall back to a value contained in a prior result

# Arguments

  - `risk_variable` : The risk-side input.
  - `prior_variable` : The prior-side input.

# Returns

  - If both inputs are `nothing` returns `nothing`.
  - If `risk_variable` is not `nothing`, returns `risk_variable`.
  - If `risk_variable` is `nothing` and `prior_variable` is not `nothing`, returns `prior_variable`.
"""
function nothing_scalar_array_selector(::Nothing, ::Nothing)
    return nothing
end
function nothing_scalar_array_selector(risk_variable::Num_ArrNum_VecScalar_DynWeights,
                                       ::Any)
    return risk_variable
end
function nothing_scalar_array_selector(::Nothing,
                                       prior_variable::Num_ArrNum_VecScalar_DynWeights)
    return prior_variable
end
"""
    risk_measure_nothing_scalar_array_view(risk_variable, prior_variable, i)

Get a view of a risk measure's risk or prior variable for index `i`.

Internal helper for slicing scalar, array, or `nothing` risk/prior variables by index. Dispatches on the types of `risk_variable` and `prior_variable`.

# Arguments

  - `risk_variable`: Risk variable (scalar, array, or `nothing`).
  - `prior_variable`: Prior variable (array or `nothing`).
  - `i`: Index or range to slice.

# Validation

  - Throws an `ArgumentError` when `risk_variable` and `prior_variable` are both `nothing`. There is then nothing to slice, and the caller named no source.

# Returns

  - Sliced or unchanged value.

# Related

  - [`port_opt_view`](@ref)
"""
function risk_measure_nothing_scalar_array_view(::Nothing, ::Nothing, i)
    return throw(ArgumentError("Both risk_variable and prior_variable are `nothing`."))
end
function risk_measure_nothing_scalar_array_view(risk_variable::Num_ArrNum, ::Any, i)
    return nothing_scalar_array_view(risk_variable, i)
end
function risk_measure_nothing_scalar_array_view(::Nothing, prior_variable::ArrNum, i)
    return nothing_scalar_array_view(prior_variable, i)
end
"""
    solver_selector(risk_solvers, slv)

Select the appropriate solver for a risk measure computation.

Returns the risk-measure-specific solver if provided, otherwise falls back to the optimiser-level solver. Neither is refused rather than answered with `nothing`: a `JuMP` model that carries no solver cannot be solved, and the refusal names that where the caller can act on it. [`sel`](@ref) never reaches the refusal, because a both-`nothing` call routes to [`nothing_scalar_array_selector`](@ref) on the operand types instead.

# Arguments

  - `risk_solvers`: Risk-measure-specific solver(s) or `nothing`.
  - `slv`: Optimiser-level solver(s) or `nothing`.

# Validation

  - Throws an `ArgumentError` when `risk_solvers` and `slv` are both `nothing`.

# Returns

  - `Slv_VecSlv`: The selected solver or solvers.

# Related

  - [`Slv_VecSlv`](@ref)
  - [`ERM`](@ref)
  - [`RRM`](@ref)
"""
function solver_selector(risk_solvers::Slv_VecSlv, ::Any)
    return risk_solvers
end
function solver_selector(::Nothing, slv::Slv_VecSlv)
    return slv
end
function solver_selector(::Nothing, ::Nothing)
    return throw(ArgumentError("Both risk_solver and prior_solver are `nothing`, cannot solve JuMP model."))
end
"""
    const DeferredQuantity = Union{...}

The dynamic half of a **Deferred Quantity**: an Estimator standing in a risk-measure slot where a prior-derived value goes. `isa(x, DeferredQuantity)` answers "is this slot deferred?" — the four moment-estimator families that compute one of the four deferrable quantities, plus [`AbstractPriorEstimator`](@ref), which computes all of them at once.

The union exists because no supertype already answers the question: [`AbstractCovarianceEstimator`](@ref) is a `StatsBase.CovarianceEstimator` and not an [`AbstractEstimator`](@ref).

# Related

  - [`MuSlot`](@ref)
  - [`SigmaSlot`](@ref)
  - [`KtSlot`](@ref)
  - [`SkSlot`](@ref)
  - [`resolve_slot`](@ref)
"""
const DeferredQuantity = Union{<:AbstractExpectedReturnsEstimator,
                               <:StatsBase.CovarianceEstimator, <:CoskewnessEstimator,
                               <:CokurtosisEstimator, <:AbstractPriorEstimator}
"""
    const MuSlot = Union{<:Num_VecNum_VecScalar, <:AbstractExpectedReturnsEstimator, <:AbstractPriorEstimator}

Field bound for an expected-returns slot: the value itself, or the Estimator that computes it. See [`DeferredQuantity`](@ref).

# Related

  - [`DeferredQuantity`](@ref)
  - [`resolve_slot`](@ref)
"""
const MuSlot = Union{<:Num_VecNum_VecScalar, <:AbstractExpectedReturnsEstimator,
                     <:AbstractPriorEstimator}
"""
    const SigmaSlot = Union{<:MatNum, <:StatsBase.CovarianceEstimator, <:AbstractPriorEstimator}

Field bound for a covariance slot: the matrix itself, or the Estimator that computes it. See [`DeferredQuantity`](@ref).

# Related

  - [`DeferredQuantity`](@ref)
  - [`resolve_slot`](@ref)
"""
const SigmaSlot = Union{<:MatNum, <:StatsBase.CovarianceEstimator, <:AbstractPriorEstimator}
"""
    const KtSlot = Union{<:MatNum, <:CokurtosisEstimator, <:AbstractPriorEstimator}

Field bound for a cokurtosis slot: the matrix itself, or the Estimator that computes it. See [`DeferredQuantity`](@ref).

# Related

  - [`DeferredQuantity`](@ref)
  - [`resolve_slot`](@ref)
"""
const KtSlot = Union{<:MatNum, <:CokurtosisEstimator, <:AbstractPriorEstimator}
"""
    const SkSlot = Union{<:MatNum, <:CoskewnessEstimator, <:AbstractPriorEstimator}

Field bound for a coskewness slot: the matrix itself, or the Estimator that computes it. See [`DeferredQuantity`](@ref).

# Related

  - [`DeferredQuantity`](@ref)
  - [`resolve_slot`](@ref)
"""
const SkSlot = Union{<:MatNum, <:CoskewnessEstimator, <:AbstractPriorEstimator}
"""
    deferred_factors(pr::AbstractPriorResult)

Return the factor returns matrix carried by prior result `pr`, or `nothing` when `pr` has no factor block. This is the only channel through which factors reach a risk-measure slot: no moment estimator takes an `F`, so a slot that must see factors has to hold an [`AbstractPriorEstimator`](@ref).

# Related

  - [`fit_deferred_quantity`](@ref)
  - [`LowOrderPrior`](@ref)
"""
function deferred_factors(pr::AbstractPriorResult)
    return isnothing(pr.fpr) ? nothing : pr.fpr.X
end
"""
    fit_deferred_quantity(dq::DeferredQuantity, pr::AbstractPriorResult)

Run a **Deferred Quantity** against the optimisation's own prior result and return what it produces: a moment estimator gives its quantity, an [`AbstractPriorEstimator`](@ref) gives a whole [`AbstractPriorResult`](@ref).

The estimator sees `pr.original_X` — the returns the **caller** supplied, sliced by any [`port_opt_view`](@ref) the measure crossed. `pr.w` is threaded through [`factory`](@ref), so a weighted prior **replaces** the estimator's own observation weights and an unweighted prior leaves them alone.

`pr.original_X` rather than `pr.X`, because the two differ on a factor route. [`FactorPrior`](@ref), [`FactorBlackLittermanPrior`](@ref) and [`AugmentedBlackLittermanPrior`](@ref) all overwrite `X` with the reconstruction `F * transpose(M) .+ transpose(b)`, which spans only the factors: it has rank `size(F, 2)`, and it carries no residual. Fitting a covariance estimator on it returns a **singular** matrix whenever there are more assets than factors, and a prior estimator in a slot could not regress it against those same factors. Off a factor route `original_X === X`, so nothing moves.

A [`CokurtosisEstimator`](@ref) gives its tensor. A [`CoskewnessEstimator`](@ref) gives the **pair** `(sk, V)` together with the matrix-processing estimator that built `V`, as a named tuple — `V` is derived from `sk` and never travels on its own. See [`coskewness_processor`](@ref).

`mean` is the centre the higher moment is taken about. [`deferred_centre`](@ref) supplies it, so that the resolved `mu` and the resolved `kt` or `sk` describe one distribution. `nothing` leaves the estimator to centre on its own `me`.

# Algorithm

 1. Rebuild `dq` against the prior's observation weights with `factory(dq, pr.w)`. The two co-moment arms bind the rebuilt estimator as `kte` and `ske`.
 2. Run the rebuilt estimator on `pr.original_X`, through the verb of its own family: `Statistics.mean` for an expected-returns estimator, `Statistics.cov` for a covariance estimator, [`cokurtosis`](@ref) for a cokurtosis estimator, [`coskewness`](@ref) for a coskewness estimator, and [`prior`](@ref) for a prior estimator, which also takes `deferred_factors(pr)`.
 3. Pass `mean` on to the two co-moment verbs, so the tensor is taken about the centre the caller named.
 4. Return the quantity the verb gave. The coskewness arm returns the named tuple `(; sk, V, skmp)` instead, because `V` comes out of the same call and [`coskewness_processor`](@ref) names the estimator that built it.

# Returns

  - The deferred quantity: a vector for an expected-returns estimator, a matrix for a covariance or a cokurtosis estimator, an [`AbstractPriorResult`](@ref) for a prior estimator, and the named tuple `(; sk, V, skmp)` for a coskewness estimator.

# Related

  - [`DeferredQuantity`](@ref)
  - [`resolve_slot`](@ref)
  - [`deferred_factors`](@ref)
  - [`deferred_centre`](@ref)
  - [`LowOrderPrior`](@ref)
  - [`_wprop`](@ref)
"""
function fit_deferred_quantity(dq::AbstractExpectedReturnsEstimator,
                               pr::AbstractPriorResult)
    return vec(Statistics.mean(factory(dq, pr.w), pr.original_X))
end
function fit_deferred_quantity(dq::StatsBase.CovarianceEstimator, pr::AbstractPriorResult)
    return Statistics.cov(factory(dq, pr.w), pr.original_X)
end
function fit_deferred_quantity(dq::AbstractPriorEstimator, pr::AbstractPriorResult)
    return prior(factory(dq, pr.w), pr.original_X, deferred_factors(pr))
end
function fit_deferred_quantity(dq::CokurtosisEstimator, pr::AbstractPriorResult;
                               mean = nothing)
    kte = factory(dq, pr.w)
    return cokurtosis(kte, pr.original_X; mean = mean)
end
function fit_deferred_quantity(dq::CoskewnessEstimator, pr::AbstractPriorResult;
                               mean = nothing)
    ske = factory(dq, pr.w)
    sk, V = coskewness(ske, pr.original_X; mean = mean)
    return (; sk = sk, V = V, skmp = coskewness_processor(ske))
end
"""
    coskewness_processor(ske::CoskewnessEstimator)

Return the matrix-processing estimator that a [`CoskewnessEstimator`](@ref) uses to build `V`, or `nothing` when the estimator names none.

`V = negative_spectral_coskewness(sk, X, mp)`, so building `V` always names a processor. When a coskewness estimator stands in a [`NegativeSkewness`](@ref) `sk` slot, **that** estimator's processor is the one that built the `V` it hands back, and the measure records it in place of its own `mp` so that a later rebuild uses the same one. This mirrors [`HighOrderPrior`](@ref)'s `skmp`.

The [`CoskewnessEstimator`](@ref) interface does not require an `mp` field, so the default answers `nothing` and the measure keeps the processor it already holds. Declare a method for an estimator that names one.

# Related

  - [`CoskewnessEstimator`](@ref)
  - [`Coskewness`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`fit_deferred_quantity`](@ref)
  - [`negative_spectral_coskewness`](@ref)
"""
function coskewness_processor(::CoskewnessEstimator)
    return nothing
end
function coskewness_processor(ske::Coskewness)
    return ske.mp
end
"""
    deferred_centre(dq, pr::AbstractPriorResult)

Return the centre that a **Deferred Quantity** in a `kt` or `sk` slot takes its moment about, or `nothing` when the estimator names none.

A higher moment is a moment **about a centre**, so the tensor and the centre are one pair of quantities out of one object. When the measure leaves `mu` unstated, the centre comes from the co-moment estimator's own `me` run on `pr.original_X`, is threaded back into the fit as `mean =`, and becomes the resolved `mu`. The measure then centres on exactly the vector its tensor was built about.

A stated `mu` wins and is threaded in its place. An [`AbstractPriorEstimator`](@ref) centres itself, so the answer is `nothing` and the centre is read off the prior result it produced instead.

The [`CokurtosisEstimator`](@ref) and [`CoskewnessEstimator`](@ref) interfaces do not require an `me` field, so the default answers `nothing`. Declare a method for an estimator that names one.

# Related

  - [`fit_deferred_quantity`](@ref)
  - [`centring_target`](@ref)
  - [`Kurtosis`](@ref)
  - [`Skewness`](@ref)
"""
function deferred_centre(::Any, ::AbstractPriorResult)
    return nothing
end
function deferred_centre(dq::Cokurtosis, pr::AbstractPriorResult)
    return vec(Statistics.mean(factory(dq, pr.w).me, pr.original_X))
end
function deferred_centre(dq::Coskewness, pr::AbstractPriorResult)
    return vec(Statistics.mean(factory(dq, pr.w).me, pr.original_X))
end
"""
    centring_target(mu)

Turn a resolved `mu` into the per-asset centre that [`cokurtosis`](@ref) and [`coskewness`](@ref) take as `mean =`.

Those two subtract the centre from the `T × N` returns matrix before they build the tensor, so the centre is a row rather than a column. A `VecScalar` contributes its scalar to every asset, which is the per-asset image of the portfolio-level target `dot(w, mu.v) + mu.s`. `nothing` leaves the estimator to centre on its own `me`.

# Related

  - [`deferred_centre`](@ref)
  - [`fit_deferred_quantity`](@ref)
  - [`calc_moment_target`](@ref)
"""
function centring_target(::Nothing)
    return nothing
end
function centring_target(mu::Number)
    return mu
end
function centring_target(mu::VecNum)
    return transpose(mu)
end
function centring_target(mu::VecScalar)
    return transpose(mu.v) .+ mu.s
end
"""
    fit_deferred_moment(dq, pr::AbstractPriorResult, centre)

Run a **Deferred Quantity** that stands in a `kt` or `sk` slot, about the centre `centre`.

A co-moment estimator takes the centre as `mean =` ([`centring_target`](@ref) puts it in the shape the estimator wants), so the tensor is built about exactly the vector the measure will resolve `mu` to.

Every other occupant centres itself and has no channel to take one, so `centre` is dropped. That is the case of an [`AbstractPriorEstimator`](@ref) in the slot: it computes its own `mu` and its own tensor about that `mu`, and the centre is read **back** off the result it produced rather than pushed into it. A `mu` the caller stated alongside such a slot still wins as the measure's centring target, and the docstring's consistency warning is what covers the gap.

# Algorithm

 1. Return `fit_deferred_quantity(dq, pr)` when `dq` is neither a [`CokurtosisEstimator`](@ref) nor a [`CoskewnessEstimator`](@ref). Such an occupant has no channel to take a centre, so `centre` is dropped.
 2. Put `centre` into the row shape a co-moment estimator wants with [`centring_target`](@ref).
 3. Run [`fit_deferred_quantity`](@ref) with that row as `mean`, and return what it gives.

# Returns

  - The moment the fit produced, taken about `centre` on the two co-moment arms.

# Related

  - [`fit_deferred_quantity`](@ref)
  - [`deferred_centre`](@ref)
  - [`centring_target`](@ref)
"""
function fit_deferred_moment(dq, pr::AbstractPriorResult, ::Any)
    return fit_deferred_quantity(dq, pr)
end
function fit_deferred_moment(dq::Union{<:CokurtosisEstimator, <:CoskewnessEstimator},
                             pr::AbstractPriorResult, centre)
    return fit_deferred_quantity(dq, pr; mean = centring_target(centre))
end
"""
    deferred_quantity(fitted, key::Symbol)

Read the quantity named by `key` off what [`fit_deferred_quantity`](@ref) produced.

A moment estimator produced the quantity itself, so it is returned and `key` is inert. A [`CoskewnessEstimator`](@ref) produced a named tuple, because `sk` and `V` come out of one call. An [`AbstractPriorEstimator`](@ref) produced a prior result, so `key` picks the one wanted from the several it computed.

# Validation

  - Throws an `ArgumentError` when `fitted` is a `NamedTuple` carrying no `key`, and when `fitted` is an [`AbstractPriorResult`](@ref) carrying no property `key`. The slot named an estimator that computes something else, and a silent `nothing` would reach the model builders as though the caller had stated none. This is the difference from [`deferred_derived_quantity`](@ref), which answers `nothing` for a key its fit does not carry.

# Returns

  - The quantity named by `key`, or the whole fit when a moment estimator produced it.

# Related

  - [`fit_deferred_quantity`](@ref)
  - [`deferred_derived_quantity`](@ref)
  - [`resolve_slot`](@ref)
"""
function deferred_quantity(fitted, ::Symbol)
    return fitted
end
function deferred_quantity(fitted::NamedTuple, key::Symbol)
    @argcheck(haskey(fitted, key),
              ArgumentError("The fit in this slot produced no `$key`, so it cannot supply it. Name an estimator that computes `$key`."))
    return fitted[key]
end
function deferred_quantity(fitted::AbstractPriorResult, key::Symbol)
    @argcheck(hasproperty(fitted, key),
              ArgumentError("A `$(typeof(fitted))` carries no `$key`, so the prior estimator in this slot cannot supply it. Name a prior estimator that computes `$key`."))
    return getproperty(fitted, key)
end
"""
    fan_out_slot(fitted, slot, key::Symbol)

Fill one slot from a fan-out fit, unless the caller stated it.

A measure that carries **two or more independently deferrable slots** takes a `pe` instead of widening each slot. [`fit_deferred_quantity`](@ref) runs that estimator **once**, and this reads one quantity per slot off the single result it produced.

The precedence is the map's: a stated slot wins, `pe` fills the rest, and a slot that neither names is left `nothing` so the consumer's own prior fallback still applies.

A measure with exactly one deferrable slot takes no `pe`. It widens that slot instead, and a derived companion — `chol` with `sigma`, `V` with `sk` — travels with it out of the same fit rather than being fanned out separately.

# Algorithm

 1. Return `slot` unchanged when it is not `nothing`. A slot the caller stated wins over the fan-out.
 2. Read the quantity named by `key` off `fitted` with [`deferred_quantity`](@ref), and return it.

# Returns

  - The stated slot, or the quantity named by `key`.

# Related

  - [`fit_deferred_quantity`](@ref)
  - [`deferred_quantity`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
function fan_out_slot(fitted, slot, key::Symbol)
    return isnothing(slot) ? deferred_quantity(fitted, key) : slot
end
"""
    deferred_derived_quantity(fitted, key::Symbol)

Read a **derived** quantity named by `key` off what [`fit_deferred_quantity`](@ref) produced — `chol`, the factorisation that travels with `sigma`; `V`, the negative spectral part that travels with `sk`; `mu`, the centre a higher moment was taken about; `skmp`, the processor that built `V`.

A prior result carries all four. A coskewness estimator produced a named tuple carrying `V` and `skmp` but no centre. Anything else produced only its own quantity, so the answer is `nothing` and the consumer keeps whatever fallback it already had. This is the difference from [`deferred_quantity`](@ref), which refuses a `key` the fit does not carry.

# Related

  - [`fit_deferred_quantity`](@ref)
  - [`deferred_quantity`](@ref)
  - [`sigma_chol_selector`](@ref)
  - [`deferred_centre`](@ref)
"""
function deferred_derived_quantity(::Any, ::Symbol)
    return nothing
end
function deferred_derived_quantity(fitted::NamedTuple, key::Symbol)
    return haskey(fitted, key) ? fitted[key] : nothing
end
function deferred_derived_quantity(fitted::AbstractPriorResult, key::Symbol)
    return hasproperty(fitted, key) ? getproperty(fitted, key) : nothing
end
"""
    resolve_slot(slot, key::Symbol, pr::AbstractPriorResult)

Resolve one risk-measure slot against prior result `pr` and return a plain value.

A slot that holds a **Deferred Quantity** is run against `pr` and the quantity named by `key` is read back. Anything else — a stated value, `nothing`, a centring strategy — is returned unchanged, so the caller can apply the ordinary prior fallback ([`sel`](@ref)) on top.

This is the whole of the third state. The other two are unchanged: `nothing` still falls back to the prior's own field, and a stated value still wins.

# Algorithm

 1. Return `slot` unchanged when it is not a [`DeferredQuantity`](@ref). A stated value, `nothing` and a centring strategy all take that arm.
 2. Run the Deferred Quantity against `pr` with [`fit_deferred_quantity`](@ref), giving `fitted`.
 3. Read the quantity named by `key` off `fitted` with [`deferred_quantity`](@ref), and return it.

# Returns

  - A plain value: the slot's own occupant, or the quantity the fit produced.

# Related

  - [`DeferredQuantity`](@ref)
  - [`fit_deferred_quantity`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`sel`](@ref)
"""
function resolve_slot(slot, ::Symbol, ::AbstractPriorResult)
    return slot
end
function resolve_slot(dq::DeferredQuantity, key::Symbol, pr::AbstractPriorResult)
    return deferred_quantity(fit_deferred_quantity(dq, pr), key)
end
"""
    deferred_slots(x)

Declare the slots of `x` that may hold a **Deferred Quantity**, as a `NamedTuple` mapping each slot's name to its current value. The default is empty: a type with no deferrable slot needs no method.

This is the declaration both consumers read. [`assert_resolved_slots`](@ref) refuses a slot that a value-level entry point cannot resolve, and [`resolve_deferred_quantities`](@ref) derives its container recursion from it. A type that names its slots here needs no forwarding method of its own.

A slot that holds a child measure is declared here too. Both consumers recurse into whatever a slot holds, so a container names its children and each child names its own slots; nothing walks fields blindly. That matters because a risk measure holds Estimators that are **not** deferred slots — a variance estimator in `ve`, an uncertainty-set estimator in `ucs` — and a blind walk would refuse them.

A type that resolves a quantity of its own — a matrix out of a covariance estimator, a tensor and the centre it was taken about out of a co-moment estimator — declares a [`resolve_deferred_quantities`](@ref) method beside this one. That method is per type because slots that travel together must be resolved together, which no derivation can know. Declaring the slots without the method is refused at the first call rather than passed over.

# Related

  - [`DeferredQuantity`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`resolve_deferred_quantities`](@ref)
"""
deferred_slots(::Any) = (;)
"""
    resolve_deferred_child(slot, pr::AbstractPriorResult, slv = nothing)

Resolve one slot that [`deferred_slots`](@ref) declared, on behalf of the derived recursion in [`resolve_deferred_quantities`](@ref).

A slot holds one of three things, and one rule covers all three. A child measure resolves through its own method. A vector of children resolves element by element, which is the rule [`factory_child`](@ref) already applies on the other path. Anything else — a stated value, `nothing`, a **Deferred Quantity** the enclosing type resolves itself — is returned unchanged.

The vector arm is bounded by the element type rather than by `AbstractArray`, so a matrix slot is a value and never a container of children.

`slv` is the effective solver, and it travels down to the child untouched. A child that carries a solver of its own settles it against this one; a child that carries none gives its rules none. So a container hands each child the same solver the enclosing measure was resolved against.

# Related

  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
  - [`factory_child`](@ref)
"""
function resolve_deferred_child(slot, pr::AbstractPriorResult, slv = nothing)
    return resolve_deferred_quantities(slot, pr, slv)
end
function resolve_deferred_child(slot::AbstractArray{<:Union{<:AbstractEstimator,
                                                            <:AbstractAlgorithm}},
                                pr::AbstractPriorResult, slv = nothing)
    return [resolve_deferred_child(s, pr, slv) for s in slot]
end
"""
    rebuild_with_slots(x, slots::NamedTuple)

Return a copy of `x` whose fields named by `slots` hold the values in `slots`.

The field list is derived from the type and the constructor is recovered from it, so nothing is written per type. The call is positional, so the inner constructor runs and every guard the type states is re-applied to the rebuilt value.

# Algorithm

 1. Read the type of `x` into `T`.
 2. Read every field of `x` into the named tuple `props`, in declaration order.
 3. Merge `slots` over `props`, so a named slot carries its new value and every other field survives.
 4. Call `T.name.wrapper` positionally on the merged values, so the inner constructor runs and re-applies every guard the type states.

# Returns

  - A value of the same type as `x`, holding the values in `slots`.

# Related

  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
"""
function rebuild_with_slots(x, slots::NamedTuple)
    T = typeof(x)
    props = NamedTuple{fieldnames(T)}(ntuple(i -> getfield(x, i), Val(fieldcount(T))))
    return T.name.wrapper(values(merge(props, slots))...)
end
"""
    assert_declared_slot_resolver(x, slots::NamedTuple)

Refuse a type that declares a deferrable slot and no way to resolve it.

`slots` is what the derived recursion produced. A **Deferred Quantity** that survives it names a type that declared the slot in [`deferred_slots`](@ref) and then wrote no [`resolve_deferred_quantities`](@ref) method, so the estimator would reach the model builders and be multiplied as though it were a matrix. ADR 0051 pairs the two declarations; this is where the pair is enforced.

# Algorithm

 1. Walk the pairs of `slots`, giving each slot's name `key` and its occupant `slot`.
 2. Refuse an occupant that is still a [`DeferredQuantity`](@ref).
 3. Return `nothing` once the walk is spent.

# Validation

  - Throws an `ArgumentError` when an entry of `slots` still holds a [`DeferredQuantity`](@ref). The message names the type, the slot and the method to declare.

# Returns

  - `nothing`.

# Related

  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
  - [`DeferredQuantity`](@ref)
"""
function assert_declared_slot_resolver(x, slots::NamedTuple)
    for (key, slot) in pairs(slots)
        @argcheck(!isa(slot, DeferredQuantity),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Deferred Quantity, a `$(nameof(typeof(slot)))`, and `$(nameof(typeof(x)))` declares no `resolve_deferred_quantities` method to fit it. The derived recursion carries a child measure's own resolution, not a quantity of the enclosing type. Declare `resolve_deferred_quantities(x::$(nameof(typeof(x))), pr::AbstractPriorResult, slv = nothing)` beside `deferred_slots`."))
    end
    return nothing
end
"""
    resolve_deferred_quantities(x, pr::AbstractPriorResult, slv = nothing)

Resolve the children that [`deferred_slots`](@ref) declared, and return `x` itself when none of them changed.

This is the derived half of the resolution rule. A container declares its children once and both entry points follow: [`factory`](@ref) reaches them through [`@fprop`](@ref), and the `JuMP` builders reach them through this method. Neither needs a forwarding method per container.

A type that resolves a quantity of its own overrides this with its own method, which is more specific. So the derivation carries container recursion alone, and never guesses how a matrix, a tensor or the centre a moment was taken about comes out of a fit.

`slv` is the effective solver, and the recursion threads it to every child. A container states no solver of its own, so it changes none: each child settles the one it was handed against the one it carries.

# Algorithm

 1. Read the slots `x` declares with [`deferred_slots`](@ref), giving `slots`.
 2. Return `x` unchanged when `slots` is empty. A type with no deferrable slot needs no method of its own.
 3. Resolve every entry of `slots` with [`resolve_deferred_child`](@ref), threading `pr` and `slv` to each, giving `resolved`.
 4. Refuse a slot the recursion left unresolved with [`assert_declared_slot_resolver`](@ref).
 5. Return `x` itself when every entry of `resolved` is the entry `slots` held, so the common case allocates nothing.
 6. Otherwise rebuild `x` around `resolved` with [`rebuild_with_slots`](@ref), and return the rebuilt value.

# Returns

  - `x` itself when no slot moved, and a rebuilt copy of `x` when one did.

# Related

  - [`deferred_slots`](@ref)
  - [`resolve_deferred_child`](@ref)
  - [`assert_declared_slot_resolver`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function resolve_deferred_quantities(x, pr::AbstractPriorResult, slv = nothing)
    slots = deferred_slots(x)
    if isempty(slots)
        return x
    end
    resolved = map(slot -> resolve_deferred_child(slot, pr, slv), slots)
    assert_declared_slot_resolver(x, resolved)
    # A container whose children resolved to themselves is returned unchanged, so the common
    # case allocates nothing and the rebuild runs only where a slot really moved.
    return if all(map(===, values(resolved), values(slots)))
        x
    else
        rebuild_with_slots(x, resolved)
    end
end
"""
    assert_resolved_slots(x)

Refuse a **Deferred Quantity** that reached a value-level entry point, which has no prior result to resolve it against.

[`expected_risk`](@ref) takes either a prior result or a plain returns matrix. Given the prior it resolves the measure through [`factory`](@ref) first. Given the matrix it cannot: that call has no `pr.w` to thread and no factor returns to reach, so resolving there would use a different rule than the settled one. So it refuses instead, naming the slot and the Estimator standing in it — without the refusal the failure lands several frames down, inside a kernel that expected a matrix.

This is the shape [`HopCount`](@ref) and [`PathLength`](@ref) already use: the consumer resolves, the kernel refuses.

The slots come from [`deferred_slots`](@ref) and the check recurses into whatever they hold, so a container is covered by its children's declarations. A slot that holds a vector of children is recursed element by element, which is the rule [`resolve_deferred_child`](@ref) applies on the resolution path. Every slot of a concretely-typed measure has a concrete field type, so the test is a type-level one and a leaf measure compiles the whole check away. A container pays one small allocation per call for the recursion into its children.

The message names both types with `nameof`, not by printing the type. A printed type carries a module prefix whenever the name is not visible from `Main`, which is the case inside an isolated test worker and inside any module that imports the package qualified. `Variance.sigma` is the path the caller wrote, and the message must read the same in every process.

# Algorithm

 1. Walk the pairs that [`deferred_slots`](@ref) declares for `x`, giving each slot's name `key` and its occupant `slot`.
 2. Refuse an occupant that holds a [`DeferredQuantity`](@ref).
 3. Recurse into the occupant, so a child measure's own slots are checked as well. A slot that holds a vector of children is walked element by element.
 4. Return `nothing` once the walk is spent.

# Validation

  - Throws an `ArgumentError` when a slot of `x`, or of any child the walk reaches, holds a [`DeferredQuantity`](@ref). The message names the slot, the Estimator standing in it and the two ways out.

# Returns

  - `nothing`.

# Related

  - [`deferred_slots`](@ref)
  - [`DeferredQuantity`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`expected_risk`](@ref)
"""
function assert_resolved_slots(x)
    for (key, slot) in pairs(deferred_slots(x))
        @argcheck(!isa(slot, DeferredQuantity),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Deferred Quantity, a `$(nameof(typeof(slot)))`, and this entry point has no prior result to resolve it against. Resolving a slot needs `pr.w` and the factor returns, which a bare returns matrix does not carry. Pass the prior result itself — `expected_risk(r, w, pr, fees)` — or resolve the measure first with `factory(r, pr)`."))
        assert_resolved_slots(slot)
    end
    return nothing
end
function assert_resolved_slots(xs::AbstractArray{<:Union{<:AbstractEstimator,
                                                         <:AbstractAlgorithm}})
    for x in xs
        assert_resolved_slots(x)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a calibrated quantity — a tail probability, a deformation parameter, an ambiguity radius, an Esfahani-Kuhn tail weight or a norm ceiling — from the data a prior result carries, so that the quantity refits whenever the sample moves.

All concrete subtypes should subtype one of the families under this root rather than the root itself, and `# Related` names them. A plain number in place of a rule is the quantity itself, exactly as it is today.

This root carries the **rules** alone. A **role**, the type that places a rule in the slot of one quantity, is an Estimator under [`AbstractCalibrationEstimator`](@ref) and subtypes nothing here, so a role placed inside another role's `alg` field is refused by the field's bound.

A **Calibration Rule** is not a [`DeferredQuantity`](@ref), and the two mechanisms stay parallel end to end. A Deferred Quantity is *fitted* and the quantity is read off the fit; a rule fits nothing, and reads the sample size and the moments the prior result already carries. A rule also sees the effective observation weights, which [`resolve_slot`](@ref) does not carry. So a rule resolves through [`resolve_calibration_slot`](@ref), is declared through [`calibration_slots`](@ref), and is refused at a value-level entry point by [`assert_calibrated_slots`](@ref).

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`DeferredQuantity`](@ref)
"""
abstract type AbstractCalibrationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Places a **Calibration Rule** in the slot of one quantity, and names the quantity the slot holds.

A role is configuration that carries an algorithm, so a role is an Estimator and the rule it carries is the Algorithm. [`AbstractOrderedWeightsArrayEstimator`](@ref) and [`AbstractOrderedWeightsArrayAlgorithm`](@ref) are the same pair, and the two calibration roots are separate for the same reason.

The separation is the whole of the role validation. Every `alg` field is bounded by a `Func_` union that names one rule family under [`AbstractCalibrationAlgorithm`](@ref), and no role subtypes that root. So a role placed inside another role's `alg` field is refused **at construction, by the bound**, and no guard method is written for it.

The root is flat, and no role carries a per-family supertype, because nothing dispatches on a family of roles. Each slot's `Num_` bound names one concrete role, and the three sites that read a role — [`resolve_calibration_slot`](@ref), [`assert_calibrated_slots`](@ref) and [`sel`](@ref) — read this root itself.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`assert_calibrated_slots`](@ref)
"""
abstract type AbstractCalibrationEstimator <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level, the tail probability that an `alpha` or a `beta` slot holds.

All concrete subtypes should subtype `AbstractSignificanceCalibrationAlgorithm`, and should be **callable**, because [`resolve_calibration_slot`](@ref) runs a rule by calling it. A plain `Function` of the same four arguments is therefore a rule as well, and needs no type at all. The family's two role types, [`SignificanceTailCalibration`](@ref) and [`SignificanceHeadCalibration`](@ref), subtype [`AbstractCalibrationEstimator`](@ref) instead: a role names the end of the distribution the slot addresses and holds the rule in its `alg` field, and the same rule serves both ends. Neither role subtypes this family, so neither is admitted by [`Func_SigCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractSignificanceCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractSignificanceCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the significance level.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the measure nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

### Returns

  - `alpha::Number`: The significance level.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
abstract type AbstractSignificanceCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a deformation parameter, the Kaniadakis ``\\kappa`` that a `kappa` slot holds.

All concrete subtypes should subtype `AbstractDeformationCalibrationAlgorithm`, and should be **callable**, on the same terms as the significance family. The family's two role types, [`DeformationTailCalibration`](@ref) and [`DeformationHeadCalibration`](@ref), subtype [`AbstractCalibrationEstimator`](@ref) instead, and neither is admitted by [`Func_DefCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractDeformationCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractDeformationCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the deformation parameter.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the measure nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

### Returns

  - `kappa::Number`: The deformation parameter.

## The series

A deformation slot sits on a measure of the return distribution and on a drawdown measure alike, and both resolve the key `:kappa`, so `key` does not say which quantity the owner prices. A rule whose answer moves with that quantity should carry a `series` field and implement a [`bind_series`](@ref) method, which each slot owner calls before it resolves the slot. A rule whose answer does not move with it needs no method, and [`EntropyBudget`](@ref) is that case.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_series`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`Func_DefCal`](@ref)
  - [`kappa_log`](@ref)
"""
abstract type AbstractDeformationCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
    const Func_SigCal = Union{<:Function, <:AbstractSignificanceCalibrationAlgorithm}

Field bound for the `alg` field of a significance role: a rule of the family, or a plain function of the same four arguments.

A rule is run by calling it, so a function and a callable struct are the same thing to [`resolve_calibration_slot`](@ref). The struct earns its keep by carrying parameters and a name that a docstring can describe; the function is the shortest way to state a one-off rule, and a closure over a caller's own data is the case that has no type.

A function carries no family, so it is admitted by both role families and the family split cannot refuse it. Only a rule that names its family is checked.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`Func_DefCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_SigCal = Union{<:Function, <:AbstractSignificanceCalibrationAlgorithm}
"""
    const Func_DefCal = Union{<:Function, <:AbstractDeformationCalibrationAlgorithm}

Field bound for the `alg` field of a deformation role: a rule of the family, or a plain function of the same four arguments. It is the counterpart of [`Func_SigCal`](@ref), and carries its reading unchanged.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_DefCal = Union{<:Function, <:AbstractDeformationCalibrationAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Places a significance rule in a slot that addresses the lower tail of the return distribution.

The role is the whole of the type: the rule itself lives in `alg`, and both ends of the distribution take the same rule. A slot bounded by [`Num_SigTailCal`](@ref) admits this role and refuses [`SignificanceHeadCalibration`](@ref), so a head rule placed in a tail slot is refused at construction rather than at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignificanceTailCalibration(;
        alg::Func_SigCal
    ) -> SignificanceTailCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`Num_SigTailCal`](@ref)
  - [`Func_SigCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct SignificanceTailCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_sig])
    """
    alg
    function SignificanceTailCalibration(alg::Func_SigCal)
        return new{typeof(alg)}(alg)
    end
end
function SignificanceTailCalibration(; alg::Func_SigCal)
    return SignificanceTailCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a significance rule in a slot that addresses the upper tail of the return distribution.

It is the counterpart of [`SignificanceTailCalibration`](@ref), and takes the same `alg` members. Every head slot in the library sits on a Range measure, so the head role never appears on a scalar measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    SignificanceHeadCalibration(;
        alg::Func_SigCal
    ) -> SignificanceHeadCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`Num_SigHeadCal`](@ref)
  - [`Func_SigCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct SignificanceHeadCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_sig])
    """
    alg
    function SignificanceHeadCalibration(alg::Func_SigCal)
        return new{typeof(alg)}(alg)
    end
end
function SignificanceHeadCalibration(; alg::Func_SigCal)
    return SignificanceHeadCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a deformation rule in a slot that addresses the lower tail of the return distribution.

It is the deformation family's counterpart of [`SignificanceTailCalibration`](@ref), and carries the same shape.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DeformationTailCalibration(;
        alg::Func_DefCal
    ) -> DeformationTailCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`Num_DefTailCal`](@ref)
  - [`Func_DefCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct DeformationTailCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_def])
    """
    alg
    function DeformationTailCalibration(alg::Func_DefCal)
        return new{typeof(alg)}(alg)
    end
end
function DeformationTailCalibration(; alg::Func_DefCal)
    return DeformationTailCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a deformation rule in a slot that addresses the upper tail of the return distribution.

It is the counterpart of [`DeformationTailCalibration`](@ref), and takes the same `alg` members.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DeformationHeadCalibration(;
        alg::Func_DefCal
    ) -> DeformationHeadCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`Num_DefHeadCal`](@ref)
  - [`Func_DefCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct DeformationHeadCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_def])
    """
    alg
    function DeformationHeadCalibration(alg::Func_DefCal)
        return new{typeof(alg)}(alg)
    end
end
function DeformationHeadCalibration(; alg::Func_DefCal)
    return DeformationHeadCalibration(alg)
end
"""
    const Num_SigTailCal = Union{<:SignificanceTailCalibration, <:Number}

Field bound for a lower-tail significance slot: the tail probability itself, or the role that computes it.

The union names one role and no other, so a head role placed in a tail slot fails the constructor's signature and is refused at construction. That is the whole of the role validation, and no guard method is written for it.

# Related

  - [`SignificanceTailCalibration`](@ref)
  - [`Num_SigHeadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_SigTailCal = Union{<:SignificanceTailCalibration, <:Number}
"""
    const Num_SigHeadCal = Union{<:SignificanceHeadCalibration, <:Number}

Field bound for an upper-tail significance slot: the tail probability itself, or the role that computes it.

# Related

  - [`SignificanceHeadCalibration`](@ref)
  - [`Num_SigTailCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_SigHeadCal = Union{<:SignificanceHeadCalibration, <:Number}
"""
    const Num_DefTailCal = Union{<:DeformationTailCalibration, <:Number}

Field bound for a lower-tail deformation slot: the deformation parameter itself, or the role that computes it.

# Related

  - [`DeformationTailCalibration`](@ref)
  - [`Num_DefHeadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_DefTailCal = Union{<:DeformationTailCalibration, <:Number}
"""
    const Num_DefHeadCal = Union{<:DeformationHeadCalibration, <:Number}

Field bound for an upper-tail deformation slot: the deformation parameter itself, or the role that computes it.

# Related

  - [`DeformationHeadCalibration`](@ref)
  - [`Num_DefTailCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_DefHeadCal = Union{<:DeformationHeadCalibration, <:Number}
"""
    resolve_calibration_slot(slot, key::Symbol, pr::AbstractPriorResult, w, slv = nothing)

Resolve one calibration slot against prior result `pr`, the effective observation weights `w` and the effective solver `slv`, and return a plain number.

A slot that holds a role type is unwrapped, and the rule in its `alg` field is **called** as `alg(key, pr, w, slv)`. So a callable rule and a plain function are the same thing here, and a rule never sees the role it was placed in. Anything else, a stated number above all, is returned unchanged.

A rule gets no portfolio. A prior result carries no portfolio weight vector, so no rule can measure a portfolio's own loss series. What it can measure is the series of each **column** of the sample, and [`bind_series`](@ref) tells it which series the slot owner prices. It does get the solver, on both of the routes that resolve a measure, so a rule may call [`ERM`](@ref) or [`RRM`](@ref). On the [`factory`](@ref) route [`@propagatable`](@ref) runs the `@cprop` selection before the resolution, so the solver is on the struct. On the `JuMP` route no selection runs, so [`set_risk_constraints!`](@ref) threads it into [`resolve_deferred_quantities`](@ref) and the owner settles it as `sel(x.slv, slv)`.

This is the parallel of [`resolve_slot`](@ref), and it is a second verb rather than a widening of the first for two reasons. `resolve_slot`'s body is `deferred_quantity(fit_deferred_quantity(dq, pr), key)`, a fit followed by an extraction, and a rule fits nothing. `resolve_slot` also carries neither `w` nor `slv`, which a rule needs. So the role types stay **out** of the [`DeferredQuantity`](@ref) union.

The caller computes `w` itself, as `sel(r.w, pr.w)`, and threads it with the measure's own `slv`. A parent that carries no observation weights of its own passes `pr.w`, and one that carries no solver leaves `slv` at its default.

**A [`TimeDependent`](@ref) reaches the host that holds the slot, and no further.** A schedule varies a *field of an estimator*, and it is consumed by [`update_time_dependent_fields`](@ref) before any prior is fitted. A rule is never standalone: it stands in a slot of a host, so the host is what a schedule swaps. Where the host is a [`JuMPOptimiser`](@ref) the four norm fields are themselves schedulable, and a schedule over them selects a rule per fold. Where the host is a risk measure the slot's own bound admits no schedule, and the caller varies the whole measure instead, through the schedulable risk-measure field of the optimiser. Both routes land in the same place, because the selection runs first and the rule then resolves against the prior of the period that was selected. A schedule *inside* a rule is therefore not a gap: it would name a fold the rule cannot see, and it would duplicate the channel the host already carries.

# Algorithm

 1. Return `slot` unchanged when it is not an [`AbstractCalibrationEstimator`](@ref). A stated number takes that arm.
 2. Read the rule out of the role's `alg` field.
 3. Call the rule as `alg(key, pr, w, slv)`, and return the number it gives. A callable struct and a plain function are the same thing here, so a rule never sees the role it was placed in.

# Arguments

  - `slot`: The slot's occupant: a number, or a role under [`AbstractCalibrationEstimator`](@ref).
  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

# Returns

  - `val::Number`: The calibrated quantity, or the stated value unchanged.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`calibration_slots`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_calibration_slot(slot, ::Symbol, ::AbstractPriorResult, ::Any,
                                  ::Any = nothing)
    return slot
end
function resolve_calibration_slot(r::AbstractCalibrationEstimator, key::Symbol,
                                  pr::AbstractPriorResult, w, slv = nothing)
    return r.alg(key, pr, w, slv)
end
"""
    calibration_slots(x)

Declare the slots of `x` that may hold a **Calibration Rule**, as a `NamedTuple` mapping each slot's name to its current value. The default is empty: a type with no calibration slot needs no method.

This is the parallel of [`deferred_slots`](@ref), and [`assert_calibrated_slots`](@ref) reads it. A type that names its slots here writes the resolution beside them, because a rule that reads a sibling slot must be resolved after that sibling and no derivation can know the order.

A slot that holds a child measure is declared here too, so a container names its children and each child names its own slots.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`assert_calibrated_slots`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`deferred_slots`](@ref)
"""
calibration_slots(::Any) = (;)
"""
    assert_calibrated_slots(x)

Refuse a **Calibration Role** that reached a value-level entry point, which has no prior result to resolve its rule against.

[`expected_risk`](@ref) takes either a prior result or a plain returns matrix. Given the prior it resolves the measure through [`factory`](@ref) first. Given the matrix it cannot: a rule reads the sample size and the moments the prior carries, and it reads the effective observation weights, none of which a bare returns matrix supplies. So it refuses instead, and names the slot, the role standing in it and the way out.

This is the shape [`assert_resolved_slots`](@ref) already uses on the Deferred-Quantity side, and the message names both types with `nameof` for the same reason: a printed type carries a module prefix wherever the name is not visible from `Main`, and the message must read the same in every process.

The slots come from [`calibration_slots`](@ref) and the check recurses into whatever they hold, so a container is covered by its children's declarations.

# Algorithm

 1. Walk the pairs that [`calibration_slots`](@ref) declares for `x`, giving each slot's name `key` and its occupant `slot`.
 2. Refuse an occupant that holds an [`AbstractCalibrationEstimator`](@ref).
 3. Recurse into the occupant, so a child measure's own slots are checked as well. A slot that holds a vector of children is walked element by element.
 4. Return `nothing` once the walk is spent.

# Validation

  - Throws an `ArgumentError` when a slot of `x`, or of any child the walk reaches, holds an [`AbstractCalibrationEstimator`](@ref). The message names the slot, the role standing in it and the two ways out.

# Returns

  - `nothing`.

# Related

  - [`calibration_slots`](@ref)
  - [`AbstractCalibrationEstimator`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`expected_risk`](@ref)
"""
function assert_calibrated_slots(x)
    for (key, slot) in pairs(calibration_slots(x))
        @argcheck(!isa(slot, AbstractCalibrationEstimator),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Calibration Role, a `$(nameof(typeof(slot)))`, and this entry point has no prior result to resolve its rule against. A rule reads the sample size, the moments and the effective observation weights, which a bare returns matrix does not carry. Pass the prior result itself — `expected_risk(r, w, pr, fees)` — or resolve the measure first with `factory(r, pr)`."))
        assert_calibrated_slots(slot)
    end
    return nothing
end
function assert_calibrated_slots(xs::AbstractArray{<:Union{<:AbstractEstimator,
                                                           <:AbstractAlgorithm}})
    for x in xs
        assert_calibrated_slots(x)
    end
    return nothing
end
"""
    mirror_role(x)

Carry the occupant of a lower-tail slot across to its upper-tail counterpart, and keep the rule.

The two ordered-weights Range types default their head slot to whatever their tail slot holds, and this verb is what carries the occupant across. A number crosses unchanged, and a tail role crosses as the head role of the same family holding the same `alg`, so the default survives the widening and no stated number moves. Every other Range type defaults its head slot to a number of its own, so a rule stated on one end of one of those is not carried to the other.

The two role families are the whole domain of the second and third methods, because a head slot's bound admits nothing else.

# Arguments

  - `x`: The lower-tail slot's occupant.

# Returns

  - `y`: The upper-tail slot's occupant.

# Related

  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
"""
function mirror_role(x::Number)
    return x
end
function mirror_role(r::SignificanceTailCalibration)
    return SignificanceHeadCalibration(; alg = r.alg)
end
function mirror_role(r::DeformationTailCalibration)
    return DeformationHeadCalibration(; alg = r.alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Names the series a calibration rule reads, so that a rule reads the quantity its slot owner prices.

A rule gets no portfolio, which [`resolve_calibration_slot`](@ref) states, so it cannot form the loss series of a portfolio that does not exist until the solver returns. What it can form is the series of each **column** of `pr.X`. This family names which one: the column itself, or the drawdown series that column carries. [`calibration_series`](@ref) is the trait a slot owner answers, and [`bind_series`](@ref) carries the answer into the rule.

The marker states the **quantity** and not the estimator, so each rule reads it on its own terms. [`HillTailDecay`](@ref) forms the series and estimates its tail index. [`RadialTailDecay`](@ref) cannot form it, because a Mahalanobis distance carries no path, so it whitens the drawdown sample instead. The three radius rules read the per-asset dispersion of the series, which [`calibration_series_dispersion`](@ref) takes, and [`TailTermParity`](@ref) reads both terms of its ratio over it.

# Related

  - [`ReturnsSeries`](@ref)
  - [`AbstractDrawdownSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_series_vec`](@ref)
  - [`bind_series`](@ref)
"""
abstract type AbstractCalibrationSeries <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Names a drawdown series, the quantity a drawdown measure prices.

A drawdown is a path functional of one series, so it is formed per column and never per row. The two members differ in how the path compounds, and they are the two series [`absolute_drawdown_vec`](@ref) and [`relative_drawdown_vec`](@ref) build.

A drawdown series is non-positive, so it carries **one end**. A rule that answers per end therefore refuses a head key under this family, and the refusal is a statement about the quantity rather than about the sample.

**What a drawdown reading says depends on the record.** A drawdown is a running functional, so its law over a finite record is not the law of one step, and a rule that reads the shape of a series reads a different shape here. A sample whose drift is strong enough for the drawdown process to settle reads **heavier** than its own returns, which is the reading a stationary drawdown carries. A sample whose drift is weak reads the range of the path over the record instead, and that range is **thinner** in the tail than the step law that made it. Both are readings of the series the measure prices over the record the measure prices it on, so a rule states the sample rather than a claim about drawdowns in general.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`series_end_sign`](@ref)
"""
abstract type AbstractDrawdownSeries <: AbstractCalibrationSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the returns themselves, the columns of `pr.X` unchanged.

It is the series every measure of the return distribution prices, and it is the default of every rule that carries a `series` field. A rule that never left this marker reads what it read before the family existed.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct ReturnsSeries <: AbstractCalibrationSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the absolute drawdown series of a column, which [`absolute_drawdown_vec`](@ref) builds.

The path is the cumulative sum of the column, and the series is the distance of that path below its own running peak. [`RelativisticDrawdownatRisk`](@ref) prices it, and [`DrawdownatRisk`](@ref) is the measure it takes its name from.

# Related

  - [`AbstractDrawdownSeries`](@ref)
  - [`RelativeDrawdownSeries`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct AbsoluteDrawdownSeries <: AbstractDrawdownSeries end
"""
$(DocStringExtensions.TYPEDEF)

Names the relative drawdown series of a column, which [`relative_drawdown_vec`](@ref) builds.

The path is the compounded value of the column, and the series is the fractional distance of that path below its own running peak. [`RelativeRelativisticDrawdownatRisk`](@ref) prices it, and [`RelativeDrawdownatRisk`](@ref) is the measure it takes its name from.

# Related

  - [`AbstractDrawdownSeries`](@ref)
  - [`AbsoluteDrawdownSeries`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`calibration_series_vec`](@ref)
"""
struct RelativeDrawdownSeries <: AbstractDrawdownSeries end
"""
    calibration_series(x)

Declare the series that `x` prices, so that a rule in one of its calibration slots reads that quantity.

The default is [`ReturnsSeries`](@ref), so a type that prices the return distribution needs no method. The two relativistic drawdown measures write one, because the series they price is not the sample they carry.

This is a trait on the **slot owner** and not a field on the rule, for the reason [`bind_norm_order`](@ref) carries a norm order rather than reading one off the rule: the quantity belongs to the measure, and a rule cannot know which measure it reached. So a marker a caller states on a rule is overwritten wherever a measure resolves it, and it serves a caller who runs the rule by hand.

# Arguments

  - `x`: The slot owner, a risk measure above all.

# Returns

  - `series::AbstractCalibrationSeries`: The series the owner prices.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_slots`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
calibration_series(::Any) = ReturnsSeries()
"""
    calibration_series_vec(series::AbstractCalibrationSeries, x::VecNum)

Return the univariate series that `series` names, built from one column of the sample.

A drawdown marker builds a fresh vector, and [`ReturnsSeries`](@ref) returns the column itself. So the returns reading allocates nothing and reads the sample it was always given.

The verb takes one column rather than the whole matrix because a drawdown is a path functional: it accumulates down a column and says nothing across a row. A caller that needs the pool walks the columns and calls this once per column.

# Arguments

  - `series`: The series marker.
  - `x`: One column of the sample.

# Returns

  - `s::VecNum`: The series the marker names.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`absolute_drawdown_vec`](@ref)
  - [`relative_drawdown_vec`](@ref)
  - [`hill_tail_index`](@ref)
"""
function calibration_series_vec(::ReturnsSeries, x::VecNum)
    return x
end
function calibration_series_vec(::AbsoluteDrawdownSeries, x::VecNum)
    return absolute_drawdown_vec(x)
end
function calibration_series_vec(::RelativeDrawdownSeries, x::VecNum)
    return relative_drawdown_vec(x)
end
"""
    calibration_series_matrix(series::AbstractCalibrationSeries, X::AbstractMatrix)

Return the sample that `series` names, one series per column of `X`.

It is [`calibration_series_vec`](@ref) over the columns, and it holds the same shape as `X`: a drawdown series carries one entry per observation, so no observation is lost and the count of a tail is the count the caller formed on `X`.

[`ReturnsSeries`](@ref) returns `X` itself, so the returns reading allocates nothing. A rule that walks the columns one at a time should call [`calibration_series_vec`](@ref) instead, and allocate one column rather than a matrix.

# Arguments

  - `series`: The series marker.
  - `X`: Returns matrix, `T × N`.

# Returns

  - `Y::AbstractMatrix`: The sample the marker names, `T × N`.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_vec`](@ref)
  - [`radial_series_inputs`](@ref)
"""
function calibration_series_matrix(::ReturnsSeries, X::AbstractMatrix)
    return X
end
function calibration_series_matrix(series::AbstractDrawdownSeries,
                                   X::AbstractMatrix{E}) where {E <: Number}
    Y = Matrix{float(E)}(undef, size(X))
    for j in axes(X, 2)
        Y[:, j] = calibration_series_vec(series, view(X, :, j))
    end
    return Y
end
"""
    calibration_series_dispersion(series::AbstractCalibrationSeries, pr::AbstractPriorResult)

Return the per-asset dispersion of the series that `series` names, one entry per column of the sample.

A [`ReturnsSeries`](@ref) reads it off `pr.sigma`, as the square root of that matrix's diagonal. That is the reading the radius family has always carried, so a prior that states a shrunk or a robust covariance matrix is read through it, and this verb leaves that untouched.

A drawdown series reads it off the **drawdown sample** instead, because a prior result states no drawdown moment. [`calibration_series_matrix`](@ref) builds the sample, and the dispersion is the sample dispersion of each of its columns. This is the reading [`radial_series_inputs`](@ref) takes a centre and a whitening factor by, and it parts from the returns reading for the same reason: `pr.sigma` is a moment of the returns, and no scaling of it states a moment of a drawdown.

A radius is a distance in the space of the scenarios the model prices, so this vector carries its units. [`ConcentrationRadius`](@ref) and [`DimensionalRateRadius`](@ref) average it into one scale, and [`DualNormRadius`](@ref) divides it by the square root of the effective sample size and takes a norm of the result.

# Arguments

  - `series`: The series the slot owner prices.
  - `pr`: Prior result the covariance matrix is read off on a returns series, and the sample on a drawdown series.

# Returns

  - `s::AbstractVector`: The per-asset dispersion of that series, `N × 1`.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref)
"""
function calibration_series_dispersion(::ReturnsSeries, pr::AbstractPriorResult)
    return sqrt.(LinearAlgebra.diag(pr.sigma))
end
function calibration_series_dispersion(series::AbstractDrawdownSeries,
                                       pr::AbstractPriorResult)
    return vec(Statistics.std(calibration_series_matrix(series, pr.X); dims = 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level from a count of observations, so that the tail keeps the same number of scenarios whatever the sample length becomes.

A stated `alpha` fixes the tail's probability. A fold half as long then leaves half as many observations in the tail, and the measure it feeds grows noisier as the sample shrinks. This rule fixes the count instead and returns `n / T`, so `ceil(alpha * T) == n` at every sample length the resolution meets.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not. Kish's effective sample size is the number of equally weighted observations that carries the same information as the weighted sample, so a weighted tail holds `n` observations in that sense rather than `n` rows. This is the only one of the three rules that reads the weights.

The rule carries no range check of its own. It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a count that produces a value outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ScenarioCount(;
        n::Number
    ) -> ScenarioCount

Keywords correspond to the struct's fields. `n` has no default, because a scenario count that suits every sample does not exist.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`RateSignificance`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct ScenarioCount <: AbstractSignificanceCalibrationAlgorithm
    """
    $(field_dict[:cal_n])
    """
    n
    function ScenarioCount(n::Number)
        return new{typeof(n)}(n)
    end
end
function ScenarioCount(; n::Number)
    return ScenarioCount(n)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the significance level that leaves `alg.n` observations in the tail of the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
\\alpha &= \\frac{n}{T_{e}}\\,,\\\\
T_{e} &= \\begin{cases}
T & \\textrm{if } w \\textrm{ is } \\texttt{nothing}\\\\
\\dfrac{\\left(\\sum\\limits_{i=1}^{T} w_{i}\\right)^{2}}{\\sum\\limits_{i=1}^{T} w_{i}^{2}} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_rm])
  - ``n``: Number of observations the tail is to hold.
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_w_i])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The count is the same for every key, so a tail slot and a head slot that carry one rule resolve to one number.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `alpha::Number`: The significance level.

# Related

  - [`ScenarioCount`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ScenarioCount)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    T = isnothing(w) ? size(pr.X, 1) : sum(w)^2 / sum(abs2, w)
    return alg.n / T
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level that shrinks with the square root of the sample length.

The tail probability is `c / sqrt(T)`, so the tail's expected count is `c * sqrt(T)`. It grows with the sample, but more slowly than the sample does, which is the rate at which a sample mean's own error falls. A longer sample therefore buys a further tail rather than only a fuller one, and [`ScenarioCount`](@ref) is the rule that buys neither.

The rule reads the raw row count, and not the effective sample size that [`ScenarioCount`](@ref) reads. The rate is a statement about the length of the record, whereas a scenario count is a statement about the observations the tail holds.

The rule carries no range check of its own, on the same terms as [`ScenarioCount`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RateSignificance(;
        c::Number = 1
    ) -> RateSignificance

Keywords correspond to the struct's fields. `c` defaults to `1`, which is the plain ``1/\\sqrt{T}`` rate.

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`ScenarioCount`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct RateSignificance <: AbstractSignificanceCalibrationAlgorithm
    """
    $(field_dict[:cal_c])
    """
    c
    function RateSignificance(c::Number)
        return new{typeof(c)}(c)
    end
end
function RateSignificance(; c::Number = 1)
    return RateSignificance(c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the significance level at the square-root rate of the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
\\alpha &= \\frac{c}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:alpha_rm])
  - ``c``: Rate coefficient.
  - $(math_dict[:T])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rate is the same for every key.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights. This rule reads the raw row count, so it ignores them.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `alpha::Number`: The significance level.

# Related

  - [`RateSignificance`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateSignificance)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    return alg.c / sqrt(size(pr.X, 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter that makes a relativistic measure spend a stated entropy budget.

[`RRM`](@ref) multiplies its dual variable by `kappa_log(inv(alpha * T), kappa)`, so that coefficient is the price the model pays for the deformation, and `target` states it directly. The rule returns the ``\\kappa`` that meets it. A stated `kappa` fixes the shape of the deformation and lets the price move with the sample; this rule fixes the price and lets the shape move.

The two quantities travel together. The rule reads its sibling `alpha`, which is why the `alpha` field exists: [`bind_alpha`](@ref) fills it with the number the slot owner resolved, and the owner's own resolution method resolves `alpha` first for that reason. A caller who runs the rule outside a measure states `alpha` instead.

The inversion is monotone, and it is solved by bisection over ``(0, 1)``. Writing ``l = \\ln(u)``, the coefficient is ``l \\sinh(\\kappa l) / (\\kappa l)``, and ``\\sinh(x)/x`` rises with ``|x|``, so the coefficient walks once from ``l`` at ``\\kappa \\to 0`` to ``\\sinh(l)`` at ``\\kappa = 1``. There is no elementary inverse of ``\\sinh(x)/x``, so the solve is a fixed sweep of 64 halvings rather than a formula.

The band the coefficient reaches moves with the sample, and a target outside it is refused at fold time. This is the one check any of the three rules carries, and it is not a range check on the quantity the rule returns — the slot owner's constructor keeps that job, as it does for the two significance rules. It states that the equation has a root at all: a target the band does not reach leaves the sweep at an end of the interval, where ``\\kappa`` is far too small or too large to be the answer to the question the caller asked and yet still inside the range the slot owner admits.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EntropyBudget(;
        target::Number,
        alpha::Option{<:Number} = nothing
    ) -> EntropyBudget

Keywords correspond to the struct's fields. `target` has no default, because the budget is the whole content of the rule. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`bind_alpha`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`RRM`](@ref)
"""
@concrete struct EntropyBudget <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_target])
    """
    target
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    function EntropyBudget(target::Number, alpha::Option{<:Number})
        return new{typeof(target), typeof(alpha)}(target, alpha)
    end
end
function EntropyBudget(; target::Number, alpha::Option{<:Number} = nothing)
    return EntropyBudget(target, alpha)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deformation parameter whose Kaniadakis logarithm meets `alg.target` on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
u &= \\frac{1}{\\alpha T}\\,,\\\\
\\ln_{\\kappa}(u) &= \\frac{u^{\\kappa} - u^{-\\kappa}}{2 \\kappa}\\,,\\\\
\\kappa &: \\ln_{\\kappa}(u) = \\tau\\,, \\quad \\kappa \\in (0,\\, 1)\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_rm])
  - ``\\tau``: Target value of the Kaniadakis logarithm.
  - ``u``: Argument of the Kaniadakis logarithm, the reciprocal of the tail's expected count.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])

# Algorithm

 1. Read the sample length `T` off `pr.X`.
 2. Form the argument `u = inv(alg.alpha * T)`, and its plain logarithm `l = log(u)`.
 3. Form the band `(lo_b, hi_b)` as the ordered pair of `l` and `(u - inv(u)) / 2`, which are the values the Kaniadakis logarithm reaches at the two ends of ``\\kappa \\in (0,\\, 1)``. `# Validation` states the refusal this band carries.
 4. Normalise the target as `target = alg.target / l`. The normalised coefficient rises once from `1`, so one comparison carries both signs of `l` and the sweep needs no sign branch.
 5. Bracket the answer with `lo = 0` and `hi = 1`.
 6. Halve the bracket 64 times. Take the midpoint `kappa` each time, and raise `lo` to it when `kappa_log(u, kappa) / l` is below `target`, or lower `hi` to it otherwise. Sixty-four halvings take the bracket below the resolution of a `Float64`.
 7. Return the midpoint of the last bracket.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The budget is the same for every key.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights. This rule reads the raw row count, so it ignores them.
  - `slv`: Effective solver. This rule needs none, because the inversion is a scalar one.

# Validation

  - `alg.alpha` must not be `nothing`.
  - `alg.target` must lie strictly between ``\\ln(u)`` and ``\\sinh(\\ln(u))``, which is the band the coefficient reaches over ``\\kappa \\in (0,\\, 1)``.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`EntropyBudget`](@ref)
  - [`bind_alpha`](@ref)
  - [`kappa_log`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::EntropyBudget)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`EntropyBudget.alpha` is `nothing`, so the rule cannot form `inv(alpha * T)`. The sibling `alpha` travels to the rule through `bind_alpha`, which the slot owner calls after it resolves `alpha`. State `alpha` on the rule itself to run it outside a measure."))
    T = size(pr.X, 1)
    u = inv(alg.alpha * T)
    l = log(u)
    lo_b, hi_b = minmax(l, (u - inv(u)) / 2)
    @argcheck(lo_b < alg.target < hi_b,
              DomainError(alg.target,
                          "`EntropyBudget.target` must lie in ($lo_b, $hi_b), the band that `kappa_log(inv(alpha * T), kappa)` reaches over `kappa` in (0, 1) at `alpha = $(alg.alpha)` and `T = $T`. No deformation parameter meets a target outside it, so the rule has nothing to return. The band moves with the sample, so a target that suits one fold need not suit another."))
    target = alg.target / l
    lo = zero(target)
    hi = one(target)
    # The normalised coefficient rises once from `1` to `sinh(l) / l` over `(0, 1)`, so one
    # comparison carries both signs of `l` and the sweep needs no sign branch. Sixty-four
    # halvings take the bracket below the resolution of a `Float64`.
    for _ in 1:64
        kappa = (lo + hi) / 2
        if kappa_log(u, kappa) / l < target
            lo = kappa
        else
            hi = kappa
        end
    end
    return (lo + hi) / 2
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Estimate the tail index of the pool of standardised values of `X`, over the worst `k` order statistics of the pool.

`series` names the series each column carries, and [`calibration_series_vec`](@ref) builds it. Every such series is centred and divided by its own sample dispersion, and `s` names the end. Hill's estimator reads the `k` of the `T N` standardised values that lie furthest into that end. [`HillTailDecay`](@ref) states the reading and the assumptions the pool carries. This verb is the estimate alone.

The pool holds `T` values per column under every marker, because a drawdown series holds one entry per observation. So the count `k` the caller formed is the count this verb reads, whatever the series is.

The element type is bound by the signature, so the pool and the sum it feeds are concrete. A rule reads `pr.X` off an [`AbstractPriorResult`](@ref), whose field types no signature states, and this is the boundary that type crosses at.

# Algorithm

 1. Read the shape of `X` into `T` and `N`, and allocate the pool `pool` of `T * N` values.
 2. Walk the columns of `X`. Build each column's series with [`calibration_series_vec`](@ref), take its sample mean `mu` and its sample dispersion `sd`, and write `-s * (col[t] - mu) / sd` into `pool`.
 3. Partially sort `pool` so that its `k + 1` smallest values sit in order at the front, and read the last of them into `vkp1`. `partialsort!` places one index alone, and it makes no promise about the rest of the vector, so the range is what puts the `k` entries the estimate reads before `vkp1`. The sign of step 2 puts the end the caller prices in the **lower** tail of the pool.
 4. Return `k` over the sum of `log(pool[i] / vkp1)` across those `k` entries. Both terms of each ratio are negative, so the ratio is one of magnitudes and the sum is Hill's with no further sign.

# Arguments

  - `series`: The series each column carries.
  - `X`: Returns matrix, `T × N`.
  - `s`: Sign of the end the estimate reads: `1` reads the gain tail and `-1` reads the loss tail.
  - `k`: Number of order statistics the estimate reads. The caller states it, and the caller keeps the floor under it.

# Validation

  - The `k + 1`-th largest pooled value must be positive.

# Returns

  - `a::Number`: The Hill estimate of the tail index of the pool.

# Related

  - [`HillTailDecay`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_vec`](@ref)
"""
function hill_tail_index(series::AbstractCalibrationSeries, X::AbstractMatrix{E},
                         s::Integer, k::Integer) where {E <: Number}
    T, N = size(X)
    pool = Vector{float(E)}(undef, T * N)
    for j in axes(X, 2)
        col = calibration_series_vec(series, view(X, :, j))
        mu = Statistics.mean(col)
        sd = Statistics.std(col; mean = mu)
        o = (j - 1) * T
        for t in 1:T
            pool[o + t] = -s * (col[t] - mu) / sd
        end
    end
    # The pool is signed so that the end being priced is the LOWER tail, and the estimate
    # reads the `k + 1` smallest. The ratio of two negatives is the ratio of their
    # magnitudes, so the sum below is Hill's with no further sign, and the message reads the
    # magnitude the caller thinks in.
    partialsort!(pool, 1:(k + 1))
    vkp1 = pool[k + 1]
    ukp1 = -vkp1
    @argcheck(ukp1 > 0,
              DomainError(ukp1,
                          "The $(k + 1)-th largest of the pooled standardised values is $ukp1, which is not positive, so there is no Hill estimate: the estimator reads `log(u_i / u_(k+1))`, which needs a positive ratio. The pool holds fewer than $(k + 1) values on the side of the mean this end prices. Lower the count `k`."))
    return k / sum(i -> log(pool[i] / vkp1), 1:k)
end
"""
    series_end_sign(series::AbstractCalibrationSeries, key::Symbol)

Return the sign of the end that `key` prices on `series`, and refuse an end the series does not have.

`:kappa_b` is the only head key, so on a returns series every other key prices the loss end. A drawdown series is non-positive and carries one end alone, so the head key names nothing on it and is refused. No drawdown Range measure ships, so the refusal is reached by a caller who runs a rule by hand.

# Arguments

  - `series`: The series the estimate is taken over.
  - `key`: Name of the slot that is being resolved.

# Validation

  - `key` must not be `:kappa_b` when `series` is an [`AbstractDrawdownSeries`](@ref).

# Returns

  - `s::Int`: `1` for the gain end and `-1` for the loss end.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`AbstractDrawdownSeries`](@ref)
  - [`HillTailDecay`](@ref)
  - [`hill_tail_index`](@ref)
"""
function series_end_sign(::ReturnsSeries, key::Symbol)
    # `:kappa_b` is the only head key, so every other key prices the loss end. One estimator
    # then serves both ends.
    return key === :kappa_b ? 1 : -1
end
function series_end_sign(series::AbstractDrawdownSeries, key::Symbol)
    @argcheck(key !== :kappa_b,
              ArgumentError("The `:kappa_b` slot prices the gain end, and a $(nameof(typeof(series))) has no gain end: a drawdown series is non-positive, so its one end is the loss end that `:kappa` and `:kappa_a` price. No drawdown Range measure ships, so a rule reached this key outside a measure. Resolve the rule under `:kappa`, or state a `ReturnsSeries`."))
    return -1
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter whose tail decays at the rate the sample's own tail decays at.

The inverse of [`kappa_log`](@ref) is the ``\\kappa``-exponential ``\\exp_{\\kappa}(x) = \\left(\\kappa x + \\sqrt{1 + \\kappa^{2} x^{2}}\\right)^{1/\\kappa}``, which goes to ``(2 \\kappa x)^{1/\\kappa}`` for large ``x``. A ``\\kappa``-deformed exponential tail is therefore a power law of index ``1/\\kappa``, and ``\\kappa`` is a reciprocal tail index. So the rule estimates the sample's tail index ``\\hat{a}`` and returns ``1/\\hat{a}``, which makes the deformation decay at the rate the sample decays at. The band the slot admits, ``(0,\\, 1)``, reads as ``\\hat{a} > 1``, and that is the condition for a finite mean. The refusal is built into the reading, on the same terms as the band of [`EntropyBudget`](@ref).

The estimator is Hill's, over the worst `k` order statistics of a **pool**. Every column of `pr.X` is centred and divided by its own sample dispersion, the standardised values are signed to the end the slot prices, and the `T * N` of them are pooled. **The rule reads `pr.X` alone, and never `pr.sigma`.** A column's dispersion comes from that column, which is the line that separates this rule from [`RadialTailDecay`](@ref). The two answer two questions: this rule asks how far one end of one column moves, and its sibling asks how far the whole cross-section moves. So this rule keeps the sign of the end and answers per end, and its sibling reads a distance and answers one number for both.

The pool carries two assumptions, and both are stated rather than hidden. The columns share one tail index after standardisation, which is what lets one estimate stand for the whole sample. The columns are also cross-correlated, so the pool holds far fewer than `T * N` independent points and the estimator's spread is wider than the nominal one of `k` points. The median of `N` per-asset estimates drops both assumptions, and pays for it in noise: at `alpha = 0.05` and `T = 250` a column leaves 12 tail points, and a Hill estimate on 12 points moves from fold to fold for no reason in the data.

`alpha` fixes the depth of the reading as well as the count. A Hill estimate reads the tail at the depth `alpha` names, and a sample whose tail approaches its power law slowly is read with a bias that falls as the depth rises: a Student-t sample read at a five per cent depth returns a ``\\kappa`` above the reciprocal of its degrees of freedom, and the same sample read at one per cent returns most of that distance. The spread of the estimate rises as the count falls, and `kmin` is the floor under that trade.

`key` says which end the slot prices, and **the answer is not the same for every key**. `:kappa` and `:kappa_a` read the loss tail, and `:kappa_b` reads the gain tail. This is the opposite of [`EntropyBudget`](@ref), whose budget is a price the model pays and is therefore one number for both ends. A tail index is a statement about a tail, and a skewed sample has two different ones, which is the whole point of the rule on a Range measure.

**`series` says which quantity the pool holds, and the slot owner states it.** A measure of the return distribution pools the columns of `pr.X`. A drawdown measure pools the per-column drawdown series of `pr.X` instead, because that is the quantity it prices, and [`bind_series`](@ref) puts the marker there at the resolution site. Nothing else in the reading moves: the same standardisation, the same count and the same estimator run over the drawdown sample. The estimate the pool then carries is the index of the drawdown series rather than of the returns, and the two are different numbers. [`AbstractDrawdownSeries`](@ref) states which way they part, and it is the record that decides.

**The per-column reading stands for the portfolio's own drawdown**, under the assumption the pool already states. The rule forms no portfolio, so it reads the drawdown series of each column and pools them, exactly as it pools the columns themselves on a returns series. The columns share one tail index after standardisation, and one map carries a column to its drawdown series, so the reading the pool gives is the reading the portfolio's own drawdown series would give. A portfolio drawdown is shallower than the average asset drawdown, and that is a statement about the **scale**, which a tail index does not read.

The count is the same count. A drawdown series holds one entry per observation, so the pool holds `T N` entries under every marker and `k = ceil(alpha * T * N)` is unchanged. No observation is dropped by the change of series.

`alpha` reaches the right series as well. A drawdown measure resolves the significance level of its own drawdown series, and the depth the rule reads is that same level under a drawdown marker.

The pool of a drawdown series carries **more dependence** than the pool of a returns series. A drawdown series is a running functional, so consecutive entries move together and one deep episode fills many of the `k` order statistics the estimate reads. The `T N` entries therefore hold far fewer independent tail points than the same count of returns, and the spread of the estimate is wider than the count states. `kmin` is stated in entries and not in independent points, so the same floor buys less here. Raise it for a drawdown owner.

The rule carries no range check on the parameter it returns. The slot owner's constructor keeps that job, as it does for every calibration rule. Its checks are statements that the estimate exists at all.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HillTailDecay(;
        kmin::Integer = 30,
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> HillTailDecay

Keywords correspond to the struct's fields. `kmin` defaults to `30`, which is the floor under the count of order statistics the estimate reads. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in. `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `kmin > 0`.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`EntropyBudget`](@ref)
  - [`hill_tail_index`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`series_end_sign`](@ref)
"""
@concrete struct HillTailDecay <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_kmin])
    """
    kmin
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    """
    $(field_dict[:cal_series])
    """
    series
    function HillTailDecay(kmin::Integer, alpha::Option{<:Number},
                           series::AbstractCalibrationSeries)
        assert_gt0(kmin, :kmin)
        return new{typeof(kmin), typeof(alpha), typeof(series)}(kmin, alpha, series)
    end
end
function HillTailDecay(; kmin::Integer = 30, alpha::Option{<:Number} = nothing,
                       series::AbstractCalibrationSeries = ReturnsSeries())
    return HillTailDecay(kmin, alpha, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deformation parameter whose reciprocal is the Hill tail index of the series that `alg.series` names, over the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
s &= \\begin{cases}
+1 & \\textrm{if the slot prices the gain end}\\\\
-1 & \\textrm{if the slot prices the loss end}
\\end{cases}\\,,\\\\
y_{tj} &= \\left(\\mathcal{S}\\left(\\boldsymbol{r}_{j}\\right)\\right)_{t}\\,,\\\\
u_{tj} &= s \\dfrac{y_{tj} - \\hat{\\mu}_{j}}{\\hat{\\sigma}_{j}}\\,,\\\\
k &= \\left\\lceil \\alpha T N \\right\\rceil\\,,\\\\
\\hat{a} &= \\dfrac{k}{\\sum\\limits_{i=1}^{k} \\ln\\left(\\dfrac{u_{(i)}}{u_{(k+1)}}\\right)}\\,,\\\\
\\kappa &= \\dfrac{1}{\\hat{a}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_rm])
  - $(math_dict[:alpha_rm]) It is the probability of the end this slot prices, so a head slot reads its own ``\\beta``.
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:r_tj])
  - ``\\boldsymbol{r}_{j}``: Column ``j`` of the sample.
  - ``\\mathcal{S}``: The series `alg.series` names, built from one column. It is the identity on a [`ReturnsSeries`](@ref), and a drawdown series on the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``y_{tj}``: Entry ``t`` of the series of column ``j``.
  - ``\\hat{\\mu}_{j}``: Sample mean of the series of column ``j``, and not of the column under a drawdown marker.
  - ``\\hat{\\sigma}_{j}``: Sample dispersion of the series of column ``j``, read off that series alone and never off ``\\hat{\\mathbf{\\Sigma}}``.
  - ``s``: Sign of the end the slot prices.
  - ``u_{tj}``: Standardised value of asset ``j`` at time ``t``, signed so that the end the slot prices is the upper tail of the pool.
  - ``u_{(i)}``: ``i``-th largest of the ``T N`` pooled values, so that ``u_{(1)} \\geq \\ldots \\geq u_{(k+1)}``.
  - ``k``: Number of order statistics the estimate reads.
  - ``\\hat{a}``: Hill estimate of the tail index of the pool.

# Algorithm

 1. Read the returns matrix off `pr` into `X`, and its element count into `np`.
 2. Take the sign `s` from `key` and `alg.series` with [`series_end_sign`](@ref). A drawdown series carries one end, so it refuses the head key rather than signing it.
 3. Form the count `k = ceil(Int, alg.alpha * np)`, the number of order statistics the estimate reads.
 4. Estimate the tail index of the pool with [`hill_tail_index`](@ref), giving `a`. The pool holds the series `alg.series` names, one entry per observation per column under every marker.
 5. Return `inv(a)`, which is the deformation parameter.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there, and its `series` field names the quantity, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. It names the end, so `:kappa` and `:kappa_a` read the loss tail and `:kappa_b` reads the gain tail. A skewed sample therefore resolves the two ends of a Range measure to two different numbers.
  - `pr`: Prior result the returns matrix is read off.
  - `w`: Effective observation weights. A tail index is a statement about the shape of a series rather than about the count of observations behind it, so this rule ignores them.
  - `slv`: Effective solver. This rule needs none, because the estimate is a closed form.

# Validation

  - `alg.alpha` must not be `nothing`.
  - `key` must name an end the series has, which [`series_end_sign`](@ref) checks.
  - `k` must be at least `alg.kmin`.
  - The pool must hold at least `k + 1` values.
  - ``u_{(k+1)}`` must be positive, which [`hill_tail_index`](@ref) checks.
  - ``\\hat{a}`` must be greater than one, which is the band ``\\kappa \\in (0,\\, 1)`` read as a tail index.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`HillTailDecay`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`EntropyBudget`](@ref)
  - [`hill_tail_index`](@ref)
  - [`kappa_log`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`series_end_sign`](@ref)
"""
function (alg::HillTailDecay)(key::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`HillTailDecay.alpha` is `nothing`, so the rule cannot form the count `k = ceil(alpha * T * N)`. The probability of the end travels to the rule through `bind_alpha`, which the slot owner calls after it resolves that end's own probability. State `alpha` on the rule itself to run it outside a measure."))
    X = pr.X
    # The sign puts the end the slot prices in the LOWER tail of the pool, and one estimator
    # then serves both ends. The series decides which ends there are to price.
    s = series_end_sign(alg.series, key)
    np = prod(size(X))
    k = ceil(Int, alg.alpha * np)
    @argcheck(k >= alg.kmin,
              DomainError(k,
                          "`HillTailDecay` reads the worst `k = ceil(alpha * T * N) = $k` of the $np pooled standardised values, and `HillTailDecay.kmin` puts the floor at $(alg.kmin). A Hill estimate over fewer order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. Lengthen the sample, widen `alpha`, or lower `kmin` and take the noise."))
    @argcheck(k + 1 <= np,
              DomainError(k,
                          "`HillTailDecay` needs $(k + 1) pooled values to form the estimate, and the pool of `T * N` holds $np. The count is `k = ceil(alpha * T * N)` at `alpha = $(alg.alpha)`, so only a probability that takes the whole sample reaches this. Lower `alpha`."))
    a = hill_tail_index(alg.series, X, s, k)
    @argcheck(a > 1,
              DomainError(a,
                          "`HillTailDecay` estimated a tail index of $a on the pool of the $(nameof(typeof(alg.series))), so `kappa = 1 / a` is $(inv(a)) and lies outside the (0, 1) the slot admits. An index of one or less is a tail with no finite mean, so no admissible deformation parameter reads it. The series is heavier-tailed than the measure can price."))
    return inv(a)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the square factor of the covariance matrix that `pr` states, so that the factor's transpose whitens a centred observation.

The factor ``\\mathbf{U}`` satisfies ``\\mathbf{U}^{\\top} \\mathbf{U} = \\hat{\\mathbf{\\Sigma}}``, which is the convention a prior result's `chol` carries and the one the variance constraint layer reads. **`chol` takes precedence over `sigma`**, which is the rule the `chol` field states, so a prior that carries a factor is whitened by that factor and never by a factorisation of `sigma`. `chol` is optional, so the common case is a `sigma` this verb factorises itself.

`chol` is checked against the length of `mu` alone, so it can be ``K \\times N`` rather than square. A rectangular factor has no inverse. A tall factor states a covariance matrix all the same, because ``\\mathbf{U}^{\\top} \\mathbf{U}`` is ``N \\times N`` at any ``K``, and the ``\\mathbf{R}`` factor of its QR factorisation is the square factor of that same matrix. A wide factor states a singular covariance matrix, and no whitening exists for it.

[`RadialTailDecay`](@ref) is the caller. The verb is separate because the precedence rule and its two refusals are a statement about a prior result rather than about a tail index.

# Algorithm

 1. Read `pr.chol` into `F`.
 2. When `F` is `nothing`, factorise `pr.sigma` with a Cholesky factorisation that reports rather than raises, and return its upper factor.
 3. Otherwise refuse a wide `F`, and take `F` itself when it is square and the `R` factor of its QR factorisation when it is tall.
 4. Refuse a zero on the diagonal of that factor, which is the rank of a triangular factor, and return the factor.

# Arguments

  - `pr`: Prior result the covariance matrix is read off.

# Validation

  - `pr.sigma` must be positive definite, when `pr.chol` is `nothing`.
  - `pr.chol` must have at least as many rows as columns.
  - The diagonal of the square factor must carry no zero.

# Returns

  - `U::AbstractMatrix`: Square factor of the covariance matrix, upper triangular whenever the prior's own factor is.

# Related

  - [`RadialTailDecay`](@ref)
  - [`radial_tail_index`](@ref)
"""
function whitening_factor(pr::AbstractPriorResult)
    F = pr.chol
    if isnothing(F)
        f = LinearAlgebra.cholesky(pr.sigma; check = false)
        @argcheck(LinearAlgebra.issuccess(f),
                  DomainError(f.info,
                              "The Cholesky factorisation of `pr.sigma` failed at pivot $(f.info), so the covariance matrix the prior states is not positive definite and no whitening of the sample exists. A shrunk or a factor-model covariance matrix reaches this. Fit a prior whose covariance matrix is positive definite, or state the factor in `pr.chol`."))
        return f.U
    end
    @argcheck(size(F, 1) >= size(F, 2),
              DimensionMismatch("`pr.chol` is $(size(F, 1)) × $(size(F, 2)), so the covariance matrix `pr.chol' * pr.chol` it states has rank at most $(size(F, 1)) over $(size(F, 2)) assets and is singular. No whitening of the sample exists. `pr.chol` takes precedence over `pr.sigma` at every consumer, so dropping to `pr.sigma` here would state something the prior does not."))
    U = size(F, 1) == size(F, 2) ? F : LinearAlgebra.UpperTriangular(LinearAlgebra.qr(F).R)
    j = findfirst(iszero, LinearAlgebra.diag(U))
    @argcheck(isnothing(j),
              DomainError(j,
                          "Entry $j of the diagonal of the square factor of `pr.chol` is zero, so the covariance matrix it states is singular and no whitening of the sample exists. The diagonal is the rank of a triangular factor, which is what `pr.chol` carries."))
    return U
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Estimate the tail index of the radial series of `X`, over the largest `k` order statistics of that series.

Every row is centred on `mu` and whitened by `U`, and the norm of the whitened row is one entry of the radial series. Hill's estimator then reads the `k` entries that lie furthest out. [`RadialTailDecay`](@ref) states the reading the series carries. This verb is the estimate alone.

The element type is bound by the signature, so the series and the sum it feeds are concrete. A rule reads `pr.X` off an [`AbstractPriorResult`](@ref), whose field types no signature states, and this is the boundary that type crosses at.

# Algorithm

 1. Solve `transpose(U) \\ transpose(X .- transpose(mu))`, giving the whitened observations as the columns of `Z`.
 2. Walk the columns of `Z`, and write the negated Euclidean norm of each into the series `d`.
 3. Partially sort `d` so that its `k + 1` smallest values sit in order at the front, and read the last of them into `vkp1`. `partialsort!` places one index alone, and it makes no promise about the rest of the vector, so the range is what puts the `k` entries the estimate reads before `vkp1`. The negation of step 2 puts the largest distances at the front.
 4. Return `k` over the sum of `log(d[i] / vkp1)` across those `k` entries. Both terms of each ratio are negative, so the ratio is one of magnitudes and the sum is Hill's with no further sign.

# Arguments

  - `X`: Returns matrix, `T × N`.
  - `mu`: Expected returns vector, `N × 1`. The rows are centred on it.
  - `U`: Square factor of the covariance matrix, which [`whitening_factor`](@ref) returns.
  - `k`: Number of order statistics the estimate reads. The caller states it, and the caller keeps the floor under it.

# Validation

  - The `k + 1`-th largest radial distance must be positive.

# Returns

  - `a::Number`: The Hill estimate of the tail index of the radial series.

# Related

  - [`RadialTailDecay`](@ref)
  - [`whitening_factor`](@ref)
"""
function radial_tail_index(X::AbstractMatrix{E}, mu::AbstractVector, U::AbstractMatrix,
                           k::Integer) where {E <: Number}
    Z = transpose(U) \ transpose(X .- transpose(mu))
    d = Vector{float(E)}(undef, size(X, 1))
    for t in axes(X, 1)
        # The series is negated so that the end of it the estimate reads is the LOWER tail,
        # which lets the selection below run without `rev = true`. `hill_tail_index` signs
        # its pool for the same reason, so the two verbs read their series the same way.
        d[t] = -LinearAlgebra.norm(view(Z, :, t))
    end
    partialsort!(d, 1:(k + 1))
    vkp1 = d[k + 1]
    dkp1 = -vkp1
    @argcheck(dkp1 > 0,
              DomainError(dkp1,
                          "The $(k + 1)-th largest radial distance is $dkp1, which is not positive, so there is no Hill estimate: the estimator reads `log(d_i / d_(k+1))`, which needs a positive ratio. The sample holds fewer than $(k + 1) observations away from `pr.mu`. Lower the count `k`."))
    return k / sum(i -> log(d[i] / vkp1), 1:k)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Kaniadakis deformation parameter whose tail decays at the rate the sample's radial series decays at.

The inverse of [`kappa_log`](@ref) is the ``\\kappa``-exponential ``\\exp_{\\kappa}(x) = \\left(\\kappa x + \\sqrt{1 + \\kappa^{2} x^{2}}\\right)^{1/\\kappa}``, which goes to ``(2 \\kappa x)^{1/\\kappa}`` for large ``x``. A ``\\kappa``-deformed exponential tail is therefore a power law of index ``1/\\kappa``, and ``\\kappa`` is a reciprocal tail index. So the rule estimates a tail index ``\\hat{a}`` and returns ``1/\\hat{a}``, which makes the deformation decay at the rate the sample decays at. The band the slot admits, ``(0,\\, 1)``, reads as ``\\hat{a} > 1``, and that is the condition for a finite mean. The refusal is built into the reading, on the same terms as the band of [`EntropyBudget`](@ref).

**The series the rule reads is radial.** A covariance matrix is a scale and ``\\kappa`` is a shape, so a rule that turns the one into the other needs a second quantity to fix the units, and the sample is the only one a prior result carries. The covariance matrix is what turns the ``T \\times N`` sample into **one** univariate series: [`whitening_factor`](@ref) gives the factor of ``\\hat{\\mathbf{\\Sigma}}``, each row is centred and whitened, and the Euclidean norm of the whitened row is that observation's Mahalanobis distance. Under an elliptical scale mixture that series carries the mixture's tail index, and the whole cross-section stands behind every entry of it. A per-column standardisation drops every off-diagonal term, so [`HillTailDecay`](@ref) cannot buy that reading and this rule cannot buy the per-column one.

**The rule is direction-blind, and that is a statement.** A distance has no sign, so one ``\\kappa`` answers both ends of a Range measure and `key` is ignored. [`HillTailDecay`](@ref) answers per end, and the difference is the whole of what the two rules say: a radial reading asks how far the cross-section moves, and a per-column reading asks how far one end of one column moves. [`mirror_role`](@ref) is therefore trivially correct for this rule, because a head role holding it returns the number a tail role holding it returns.

The series holds `T` entries where the pool of [`HillTailDecay`](@ref) holds `T N`, so this rule reads **fewer** tail points from the same sample: `alpha = 0.05` at `T = 250` leaves 12 of them. `kmin` is the floor under that count, and it is stated in the same units as its sibling's, so the floor binds harder here. A count below it is refused rather than estimated. The points the series does hold are one per observation rather than `N` per observation, so they carry none of the cross-correlation the pool carries.

**`series` says which sample the rows are read off, and the slot owner states it.** A measure of the return distribution whitens the rows of `pr.X`. A drawdown measure whitens the rows of the **drawdown sample**: [`calibration_series_matrix`](@ref) turns each column into the drawdown series it carries, and the rows of that sample are whitened and normed on the same terms. So the radial reading itself is unchanged, and only the sample it reads moves. The distance is then the depth of the whole cross-section in drawdown at one date, where on a returns series it is the distance the whole cross-section moved.

**A drawdown sample carries its own moments, and the prior states none of them.** `pr.mu` and `pr.sigma` are the moments of the returns, and no scaling of them states the moments of a drawdown. So under a drawdown marker the centre is the column means of the drawdown sample and the factor is the Cholesky factor of its covariance matrix, both taken off that sample. [`radial_series_inputs`](@ref) is where the two readings part, and the precedence of `pr.chol` over `pr.sigma` therefore governs the returns reading alone.

A third reading of ``\\kappa`` exists and this rule does not take it. The excess kurtosis ``g`` of a Student-t gives ``\\nu = 4 + 6/g``, and ``\\kappa = 1/\\nu`` follows. It reads the whole sample rather than the tail, so it is steady where a Hill estimate is noisy, and it pays for that with an assumption about the shape of the whole distribution rather than of its tail.

The rule carries no range check on the parameter it returns. The slot owner's constructor keeps that job, as it does for every calibration rule. Its checks are statements that the estimate exists at all. A near-singular covariance matrix is not among them: the whitening then reads the sample's smallest eigen-direction, the radial series follows it, and that is the covariance matrix speaking rather than a defect.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RadialTailDecay(;
        kmin::Integer = 30,
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> RadialTailDecay

Keywords correspond to the struct's fields. `kmin` defaults to `30`, which is the floor under the count of order statistics the estimate reads. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in. `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `kmin > 0`.

# Related

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`radial_tail_index`](@ref)
  - [`whitening_factor`](@ref)
"""
@concrete struct RadialTailDecay <: AbstractDeformationCalibrationAlgorithm
    """
    $(field_dict[:cal_kmin_rad])
    """
    kmin
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    """
    $(field_dict[:cal_series_rad])
    """
    series
    function RadialTailDecay(kmin::Integer, alpha::Option{<:Number},
                             series::AbstractCalibrationSeries)
        assert_gt0(kmin, :kmin)
        return new{typeof(kmin), typeof(alpha), typeof(series)}(kmin, alpha, series)
    end
end
function RadialTailDecay(; kmin::Integer = 30, alpha::Option{<:Number} = nothing,
                         series::AbstractCalibrationSeries = ReturnsSeries())
    return RadialTailDecay(kmin, alpha, series)
end
"""
    radial_series_inputs(series::AbstractCalibrationSeries, pr::AbstractPriorResult)

Return the sample, the centre and the whitening factor that the radial series of `series` is built from.

A [`ReturnsSeries`](@ref) reads the three off the prior result: `pr.X`, `pr.mu` and the factor [`whitening_factor`](@ref) takes off `pr.chol` or `pr.sigma`. That is the reading [`RadialTailDecay`](@ref) has always carried, and this verb leaves it untouched.

A drawdown series reads the three off the **drawdown sample** instead, because a prior result states no drawdown moment. [`calibration_series_matrix`](@ref) builds the sample, the centre is the column means of that sample, and the factor is the Cholesky factor of its covariance matrix. So the whole reading moves to the quantity the measure prices, and `pr.mu` and `pr.sigma` reach nothing: they are the moments of the returns, and no scaling of them states the moments of a drawdown.

# Arguments

  - `series`: The series the slot owner prices.
  - `pr`: Prior result the sample and, on a returns series, the moments are read off.

# Validation

  - The covariance matrix of the drawdown sample must be positive definite. A column with no movement, and two columns with one path between them, are what reach this.

# Returns

  - `Y::AbstractMatrix`: The sample the marker names, `T × N`.
  - `mu::AbstractVector`: The centre of that sample, `N × 1`.
  - `U::AbstractMatrix`: Square factor of its covariance matrix.

# Related

  - [`RadialTailDecay`](@ref)
  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`radial_tail_index`](@ref)
  - [`whitening_factor`](@ref)
"""
function radial_series_inputs(::ReturnsSeries, pr::AbstractPriorResult)
    return pr.X, pr.mu, whitening_factor(pr)
end
function radial_series_inputs(series::AbstractDrawdownSeries, pr::AbstractPriorResult)
    Y = calibration_series_matrix(series, pr.X)
    mu = vec(Statistics.mean(Y; dims = 1))
    # A prior result states the moments of the RETURNS, so the moments of the drawdown
    # sample come off that sample. `whitening_factor` is not the verb here: its precedence
    # rule and its refusals are statements about `pr.chol` and `pr.sigma`.
    f = LinearAlgebra.cholesky(Statistics.cov(Y); check = false)
    @argcheck(LinearAlgebra.issuccess(f),
              DomainError(f.info,
                          "The Cholesky factorisation of the covariance matrix of the $(nameof(typeof(series))) sample failed at pivot $(f.info), so that matrix is not positive definite and no whitening of the drawdown sample exists. A column that never moves has a drawdown series of zeros, and two columns with one path between them state one drawdown series twice. Drop the column, or state `kappa` on the slot."))
    return Y, mu, f.U
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the deformation parameter whose reciprocal is the Hill tail index of the radial series of the sample that `alg.series` names.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{y}_{t} &= \\mathcal{S}\\left(\\boldsymbol{x}_{t}\\right)\\,,\\\\
\\hat{\\mathbf{\\Sigma}}_{y} &= \\mathbf{U}^{\\top} \\mathbf{U}\\,,\\\\
\\boldsymbol{z}_{t} &= \\mathbf{U}^{-\\top} \\left(\\boldsymbol{y}_{t} - \\hat{\\boldsymbol{\\mu}}_{y}\\right)\\,,\\\\
d_{t} &= \\left\\lVert \\boldsymbol{z}_{t} \\right\\rVert_{2}\\,,\\\\
k &= \\left\\lceil \\alpha T \\right\\rceil\\,,\\\\
\\hat{a} &= \\dfrac{k}{\\sum\\limits_{i=1}^{k} \\ln\\left(\\dfrac{d_{(i)}}{d_{(k+1)}}\\right)}\\,,\\\\
\\kappa &= \\dfrac{1}{\\hat{a}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:kappa_rm])
  - $(math_dict[:alpha_rm]) It is the probability of the end the slot prices, and the count it fixes is the same for both ends because the series has no sign.
  - $(math_dict[:T])
  - $(math_dict[:x_t_obs])
  - ``\\mathcal{S}``: The sample `alg.series` names, built one column at a time. It is the identity on a [`ReturnsSeries`](@ref), and a drawdown series on the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``\\boldsymbol{y}_{t}``: Row ``t`` of that sample.
  - ``\\hat{\\boldsymbol{\\mu}}_{y}``: Centre of that sample. It is `pr.mu` on a returns series, and the column means of the drawdown sample on a drawdown series.
  - ``\\hat{\\mathbf{\\Sigma}}_{y}``: Covariance matrix of that sample. It is `pr.sigma` on a returns series, and the covariance matrix of the drawdown sample on a drawdown series.
  - ``\\mathbf{U}``: Square factor of ``\\hat{\\mathbf{\\Sigma}}_{y}``, which is `pr.chol` when the prior carries one and the slot prices the returns.
  - ``\\boldsymbol{z}_{t}``: Whitened observation ``t``.
  - ``d_{t}``: Mahalanobis distance of observation ``t``, the ``t``-th entry of the radial series.
  - ``d_{(i)}``: ``i``-th largest entry of the radial series, so that ``d_{(1)} \\geq \\ldots \\geq d_{(k+1)}``.
  - ``k``: Number of order statistics the estimate reads.
  - ``\\hat{a}``: Hill estimate of the tail index of the radial series.

# Algorithm

 1. Read the returns matrix off `pr` into `X`, and its row count into `T`.
 2. Form the count `k = ceil(Int, alg.alpha * T)`, the number of order statistics the estimate reads. A drawdown series holds one entry per observation, so the count is the same count under every marker.
 3. Take the sample, its centre and its whitening factor with [`radial_series_inputs`](@ref).
 4. Estimate the tail index of the radial series with [`radial_tail_index`](@ref), giving `a`.
 5. Return `inv(a)`, which is the deformation parameter.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there, and its `series` field names the quantity, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The radial series has no sign, so this rule reads no end from it and returns one number for every key.
  - `pr`: Prior result the returns matrix, the expected returns vector and the covariance matrix are read off.
  - `w`: Effective observation weights. A tail index is a statement about the shape of a series rather than about the count of observations behind it, so this rule ignores them.
  - `slv`: Effective solver. This rule needs none, because the estimate is a closed form.

# Validation

  - `alg.alpha` must not be `nothing`.
  - `k` must be at least `alg.kmin`.
  - The series must hold at least `k + 1` entries.
  - The sample must state a whitening, which [`radial_series_inputs`](@ref) checks on both readings.
  - ``d_{(k+1)}`` must be positive, which [`radial_tail_index`](@ref) checks.
  - ``\\hat{a}`` must be greater than one, which is the band ``\\kappa \\in (0,\\, 1)`` read as a tail index.

# Returns

  - `kappa::Number`: The deformation parameter.

# Related

  - [`RadialTailDecay`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`kappa_log`](@ref)
  - [`radial_series_inputs`](@ref)
  - [`radial_tail_index`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`whitening_factor`](@ref)
"""
function (alg::RadialTailDecay)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`RadialTailDecay.alpha` is `nothing`, so the rule cannot form the count `k = ceil(alpha * T)`. The probability of the end travels to the rule through `bind_alpha`, which the slot owner calls after it resolves that end's own probability. State `alpha` on the rule itself to run it outside a measure."))
    X = pr.X
    T = size(X, 1)
    k = ceil(Int, alg.alpha * T)
    @argcheck(k >= alg.kmin,
              DomainError(k,
                          "`RadialTailDecay` reads the largest `k = ceil(alpha * T) = $k` of the $T radial distances, and `RadialTailDecay.kmin` puts the floor at $(alg.kmin). A Hill estimate over fewer order statistics moves from fold to fold for no reason in the data, and the deformation parameter moves with it. The radial series holds one entry per observation where the pool of `HillTailDecay` holds `N`, so the same floor binds harder here. Lengthen the sample, widen `alpha`, or lower `kmin` and take the noise."))
    @argcheck(k + 1 <= T,
              DomainError(k,
                          "`RadialTailDecay` needs $(k + 1) radial distances to form the estimate, and the series holds one per observation, which is $T. The count is `k = ceil(alpha * T)` at `alpha = $(alg.alpha)`, so only a probability that takes the whole sample reaches this. Lower `alpha`."))
    Y, mu, U = radial_series_inputs(alg.series, pr)
    a = radial_tail_index(Y, mu, U, k)
    @argcheck(a > 1,
              DomainError(a,
                          "`RadialTailDecay` estimated a tail index of $a on the radial series of the $(nameof(typeof(alg.series))), so `kappa = 1 / a` is $(inv(a)) and lies outside the (0, 1) the slot admits. An index of one or less is a tail with no finite mean, so no admissible deformation parameter reads it. The series is heavier-tailed than the measure can price."))
    return inv(a)
end
"""
    bind_alpha(slot, alpha::Number)

Hand a resolved `alpha` to the rule that reads it, and return the slot's occupant with the number in place.

Two pairs **travel** through this verb. `alpha` and `kappa`: [`EntropyBudget`](@ref), [`HillTailDecay`](@ref) and [`RadialTailDecay`](@ref) each read the significance level of a sibling slot. `alpha` and `l`: [`TailTermParity`](@ref) prices a tail term at the measure's own significance level. [`resolve_calibration_slot`](@ref) carries a `Symbol` and no number, so the number travels through the rule itself. The slot owner's own resolution method resolves `alpha` first, calls this verb on the slot that reads it, and resolves the result:

```julia
alpha = resolve_calibration_slot(x.alpha, :alpha, pr, w, slv)
kappa = resolve_calibration_slot(bind_alpha(x.kappa, alpha), :kappa, pr, w, slv)
```

The default is the identity, so a stated number, a plain function and a rule that reads no sibling all pass through untouched. A travelling role is rebuilt around the bound rule, which is what lets the verb take the slot rather than the rule the caller has to unwrap first. The significance family and the radius family need no method, because no rule of either reads a sibling and the identity is already the right answer for both.

# Arguments

  - `slot`: The slot's occupant: a number, a travelling role, or a rule.
  - `alpha`: The sibling slot's resolved significance level.

# Returns

  - `bound`: The occupant, with `alpha` in place wherever a rule reads it.

# Related

  - [`EntropyBudget`](@ref)
  - [`HillTailDecay`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`TailTermParity`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function bind_alpha(slot, ::Number)
    return slot
end
function bind_alpha(r::DeformationTailCalibration, alpha::Number)
    return DeformationTailCalibration(; alg = bind_alpha(r.alg, alpha))
end
function bind_alpha(r::DeformationHeadCalibration, alpha::Number)
    return DeformationHeadCalibration(; alg = bind_alpha(r.alg, alpha))
end
function bind_alpha(alg::EntropyBudget, alpha::Number)
    return EntropyBudget(; target = alg.target, alpha = alpha)
end
function bind_alpha(alg::HillTailDecay, alpha::Number)
    return HillTailDecay(; kmin = alg.kmin, alpha = alpha, series = alg.series)
end
function bind_alpha(alg::RadialTailDecay, alpha::Number)
    return RadialTailDecay(; kmin = alg.kmin, alpha = alpha, series = alg.series)
end
"""
    bind_series(slot, series::AbstractCalibrationSeries)

Hand the series a slot owner prices to the rule that reads it, and return the slot's occupant with the marker in place.

A rule gets a prior result and no portfolio, so the quantity it can read is the sample the prior carries. Which series of that sample is the right one is a property of the **owner**: [`RelativisticValueatRisk`](@ref) prices the returns and [`RelativisticDrawdownatRisk`](@ref) prices a drawdown series of them. [`resolve_calibration_slot`](@ref) carries a `Symbol` and no marker, and the key `:kappa` serves both owners, so the marker travels through the rule itself. This is the shape [`bind_alpha`](@ref) uses to carry a significance level, and the shape [`bind_norm_order`](@ref) uses to carry a norm order.

The owner's series **wins**, on the terms [`bind_norm_order`](@ref) states. A rule that already carries a marker has it replaced, because the quantity belongs to the measure and a rule cannot know which measure it reached. So a stated `series` serves a caller who runs the rule by hand, and nothing else.

The default is the identity, so a stated number, a travelling role, a plain function and a rule that reads no series all pass through untouched. [`EntropyBudget`](@ref) needs no method: it reads the sample length and the sibling `alpha`, and neither moves with the series. The significance and norm-ceiling families need none either. A significance level is a probability and carries no units, and a norm ceiling is a bound on the weight vector rather than on the sample.

**The radius and tail-weight families do need one, and it is the units that say so.** A radius is a distance in the space of the scenarios the model prices, and a tail weight is the exchange rate between two terms of a loss, so both are read off a scale of that quantity. Under [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) the quantity is a per-asset drawdown, which is what the transport cost of its own programme is measured against.

The slot owner's own resolution method calls this beside [`bind_alpha`](@ref):

```julia
alpha = resolve_calibration_slot(x.alpha, :alpha, pr, w, slv)
kappa = resolve_calibration_slot(bind_series(bind_alpha(x.kappa, alpha),
                                             calibration_series(x)), :kappa, pr, w, slv)
```

# Arguments

  - `slot`: The slot's occupant: a number, a travelling role, or a rule.
  - `series`: The series the slot owner prices, which [`calibration_series`](@ref) states.

# Returns

  - `bound`: The occupant, with `series` in place wherever a rule reads one.

# Related

  - [`AbstractCalibrationSeries`](@ref)
  - [`calibration_series`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_norm_order`](@ref)
  - [`HillTailDecay`](@ref)
  - [`RadialTailDecay`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref)
  - [`TailTermParity`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function bind_series(slot, ::AbstractCalibrationSeries)
    return slot
end
function bind_series(r::DeformationTailCalibration, series::AbstractCalibrationSeries)
    return DeformationTailCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(r::DeformationHeadCalibration, series::AbstractCalibrationSeries)
    return DeformationHeadCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(alg::HillTailDecay, series::AbstractCalibrationSeries)
    return HillTailDecay(; kmin = alg.kmin, alpha = alg.alpha, series = series)
end
function bind_series(alg::RadialTailDecay, series::AbstractCalibrationSeries)
    return RadialTailDecay(; kmin = alg.kmin, alpha = alg.alpha, series = series)
end
# The two ambiguity roles and their rules stand below the deformation pair for the reason
# the two `bind_alpha` methods above them do: the types are declared further down the file,
# and the role is rebuilt around the bound rule on the same terms.
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the data a prior result carries, so that the radius refits whenever the sample moves.

An **Ambiguity Radius** is the radius of the ball of probability measures the model prices, and it is the coefficient of an un-squared norm penalty on the weight vector. It is neither a tail probability nor a deformation parameter, so it takes its own family beside the two of [`AbstractSignificanceCalibrationAlgorithm`](@ref) and [`AbstractDeformationCalibrationAlgorithm`](@ref), under the same root.

All concrete subtypes should subtype `AbstractAmbiguityRadiusCalibrationAlgorithm`, and should be **callable**, on the same terms as the two other families. The family's role type, [`AmbiguityRadiusCalibration`](@ref), subtypes [`AbstractCalibrationEstimator`](@ref) instead, and is not admitted by [`Func_AmbRadCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractAmbiguityRadiusCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractAmbiguityRadiusCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the ambiguity radius.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the owner carries none.

### Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
"""
abstract type AbstractAmbiguityRadiusCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes an Esfahani-Kuhn tail weight, the weight that the `l` slot of a distributionally robust measure holds.

The tail weight scales the tail term of the Esfahani-Kuhn loss, and the mean term is not scaled by it. It is a different quantity from the radius, so it takes a different family and a different role: a radius rule placed in a tail-weight slot is refused at construction. The family's role type, [`AmbiguityTailWeightCalibration`](@ref), subtypes [`AbstractCalibrationEstimator`](@ref) rather than this family, and is not admitted by [`Func_AmbTwtCal`](@ref).

[`TailTermParity`](@ref) is the rule the family ships. A stated tail weight is dimensionless and is not scale-free in the sample, so one number is a different trade-off at every sampling frequency; the rule reads the sample's own units and leaves the preference in the caller's `ratio`. [`Func_AmbTwtCal`](@ref) admits a plain function beside it, so a caller's own rule needs no type.

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractAmbiguityTailWeightCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractAmbiguityTailWeightCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the tail weight.

### Arguments

  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads its sample size and moments off.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the owner carries none.

### Returns

  - `l::Number`: The tail weight.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Func_AmbTwtCal`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`TailTermParity`](@ref)
"""
abstract type AbstractAmbiguityTailWeightCalibrationAlgorithm <:
              AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a norm ceiling from the data a prior result carries, so that the ceiling refits whenever the universe moves.

A **Norm Ceiling** is an upper bound on a norm of the weight vector, the quantity the `l2c`, `lpc` and `linfc` slots of [`JuMPOptimiser`](@ref) hold. It is not an **Ambiguity Radius**. A radius is the coefficient of a norm penalty in the objective, and a ceiling bounds that norm in a constraint. The reciprocal of a ceiling is a floor on the effective number of assets, which is a diversification statement rather than a statement about the set of measures the model prices. So the quantity takes its own family beside the others, under the same root.

All concrete subtypes should subtype `AbstractNormCeilingCalibrationAlgorithm`, and should be **callable**, on the same terms as the other families. The family's role type, [`NormCeilingCalibration`](@ref), subtypes [`AbstractCalibrationEstimator`](@ref) instead, and is not admitted by [`Func_NormCeilCal`](@ref).

# Interfaces

In order to implement a new concrete type that works seamlessly with the library, subtype `AbstractNormCeilingCalibrationAlgorithm` and implement the following method:

## The functor

  - `(alg::AbstractNormCeilingCalibrationAlgorithm)(key::Symbol, pr::AbstractPriorResult, w, slv) -> Number`: Returns the norm ceiling.

### Arguments

  - `key`: Name of the slot that is being resolved: `:l2c`, `:lpc` or `:linfc`.
  - `pr`: Prior result the rule reads its asset count and sample size off.
  - `w`: Effective observation weights, or `nothing` when neither the owner nor the prior names any.
  - `slv`: Effective solver, or `nothing` when the owner carries none.

### Returns

  - `val::Number`: The norm ceiling.

## The norm order

A ceiling is read against one norm order, and that order belongs to the constraint rather than to the rule. A rule that needs the order should implement a [`bind_norm_order`](@ref) method, which each constraint site calls before it resolves the slot. A rule that needs no order needs no method.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`NormCeilingCalibration`](@ref)
  - [`Func_NormCeilCal`](@ref)
  - [`bind_norm_order`](@ref)
  - [`EffectiveAssetFloor`](@ref)
"""
abstract type AbstractNormCeilingCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
    const Func_AmbRadCal = Union{<:Function, <:AbstractAmbiguityRadiusCalibrationAlgorithm}

Field bound for the `alg` field of an ambiguity-radius role: a rule of the family, or a plain function of the same four arguments. It is the counterpart of [`Func_SigCal`](@ref), and carries its reading unchanged.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`Func_AmbTwtCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_AmbRadCal = Union{<:Function, <:AbstractAmbiguityRadiusCalibrationAlgorithm}
"""
    const Func_AmbTwtCal = Union{<:Function,
                                 <:AbstractAmbiguityTailWeightCalibrationAlgorithm}

Field bound for the `alg` field of an ambiguity-tail-weight role: a rule of the family, or a plain function of the same four arguments. [`TailTermParity`](@ref) is the rule the family ships, and the plain function carries a caller's own.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`TailTermParity`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_AmbTwtCal = Union{<:Function, <:AbstractAmbiguityTailWeightCalibrationAlgorithm}
"""
    const Func_NormCeilCal = Union{<:Function,
                                   <:AbstractNormCeilingCalibrationAlgorithm}

Field bound for the `alg` field of a norm-ceiling role: a rule of the family, or a plain function of the same four arguments. It is the counterpart of [`Func_AmbRadCal`](@ref), and carries its reading unchanged.

# Related

  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_NormCeilCal = Union{<:Function, <:AbstractNormCeilingCalibrationAlgorithm}
"""
$(DocStringExtensions.TYPEDEF)

Places an ambiguity-radius rule in a slot that holds the radius of the ball the model prices.

The role is the whole of the type: the rule itself lives in `alg`. A radius has no lower and upper end to name, so the family carries one role rather than the two that a tail probability carries, and [`mirror_role`](@ref) needs no method for it.

A slot bounded by [`Num_AmbRadCal`](@ref) admits this role and refuses [`AmbiguityTailWeightCalibration`](@ref), so a tail-weight rule placed in a radius slot is refused at construction rather than at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AmbiguityRadiusCalibration(;
        alg::Func_AmbRadCal
    ) -> AmbiguityRadiusCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
"""
@concrete struct AmbiguityRadiusCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_amb_rad])
    """
    alg
    function AmbiguityRadiusCalibration(alg::Func_AmbRadCal)
        return new{typeof(alg)}(alg)
    end
end
function AmbiguityRadiusCalibration(; alg::Func_AmbRadCal)
    return AmbiguityRadiusCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a tail-weight rule in a slot that holds the weight of the tail term of an Esfahani-Kuhn loss.

It is the counterpart of [`AmbiguityRadiusCalibration`](@ref), and carries the same shape. Its `alg` holds [`TailTermParity`](@ref) or a caller's own function.

The role **travels**, and the radius role does not. A tail weight prices a tail at the measure's own significance level, so [`bind_alpha`](@ref) rebuilds the role around the bound rule before the slot is resolved, on the same terms as the two deformation roles.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AmbiguityTailWeightCalibration(;
        alg::Func_AmbTwtCal
    ) -> AmbiguityTailWeightCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`Func_AmbTwtCal`](@ref)
  - [`TailTermParity`](@ref)
  - [`bind_alpha`](@ref)
"""
@concrete struct AmbiguityTailWeightCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_amb_twt])
    """
    alg
    function AmbiguityTailWeightCalibration(alg::Func_AmbTwtCal)
        return new{typeof(alg)}(alg)
    end
end
function AmbiguityTailWeightCalibration(; alg::Func_AmbTwtCal)
    return AmbiguityTailWeightCalibration(alg)
end
"""
$(DocStringExtensions.TYPEDEF)

Places a norm-ceiling rule in a slot that bounds a norm of the weight vector from above.

A ceiling is a different quantity from a radius, so it takes a different role: a radius rule placed in a ceiling slot is refused at construction, and a ceiling rule placed in a radius slot is refused the same way. [`AbstractNormCeilingCalibrationAlgorithm`](@ref) states the difference between the two quantities.

The role is the whole of the type: the rule itself lives in `alg`. A ceiling has no lower and upper end to name, so the family carries one role rather than two, and [`mirror_role`](@ref) needs no method for it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    NormCeilingCalibration(;
        alg::Func_NormCeilCal
    ) -> NormCeilingCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractCalibrationEstimator`](@ref)
  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`Func_NormCeilCal`](@ref)
  - [`EffectiveAssetFloor`](@ref)
"""
@concrete struct NormCeilingCalibration <: AbstractCalibrationEstimator
    """
    $(field_dict[:cal_alg_norm_ceil])
    """
    alg
    function NormCeilingCalibration(alg::Func_NormCeilCal)
        return new{typeof(alg)}(alg)
    end
end
function NormCeilingCalibration(; alg::Func_NormCeilCal)
    return NormCeilingCalibration(alg)
end
"""
    const Num_AmbRadCal = Union{<:AmbiguityRadiusCalibration, <:Number}

Field bound for an ambiguity-radius slot: the radius itself, or the role that computes it.

The union names one role and no other, so a tail-weight role placed in a radius slot fails the constructor's signature and is refused at construction. That is the whole of the role validation, and no guard method is written for it.

# Related

  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_AmbRadCal = Union{<:AmbiguityRadiusCalibration, <:Number}
"""
    const Num_AmbTwtCal = Union{<:AmbiguityTailWeightCalibration, <:Number}

Field bound for an ambiguity-tail-weight slot: the tail weight itself, or the role that computes it.

# Related

  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_AmbTwtCal = Union{<:AmbiguityTailWeightCalibration, <:Number}
"""
    const Num_NormCeilCal = Union{<:NormCeilingCalibration, <:Number}

Field bound for a norm-ceiling slot: the ceiling itself, or the role that computes it.

The union names one role and no other, so a radius role placed in a ceiling slot fails the constructor's signature and is refused at construction. That is the whole of the role validation, and no guard method is written for it.

# Related

  - [`NormCeilingCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Num_AmbRadNormCeilCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Num_NormCeilCal = Union{<:NormCeilingCalibration, <:Number}
"""
    const Num_AmbRadNormCeilCal = Union{<:AmbiguityRadiusCalibration,
                                        <:NormCeilingCalibration, <:Number}

Field bound for the one slot the library reads as two quantities, the `val` field of [`LpRegularisation`](@ref).

That estimator is a penalty in the `lp` field of [`JuMPOptimiser`](@ref) and a norm constraint in its `lpc` field, so `val` is an ambiguity radius on one route and a norm ceiling on the other. One field cannot carry two bounds, so this bound admits both roles and each route refuses the role that has no reading on it. It is the only slot in the library whose role is settled after construction rather than by its bound.

# Related

  - [`LpRegularisation`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`assert_penalty_coefficient_role`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
"""
const Num_AmbRadNormCeilCal = Union{<:AmbiguityRadiusCalibration, <:NormCeilingCalibration,
                                    <:Number}
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the concentration of measure, so that the ball shrinks as the sample grows.

The radius is the Blanchet-Kang-Murthy form, a scale in the units of the series the slot owner prices times the square root of a chi-squared quantile over the sample size. The chi-squared factor is dimensionless and grows with the number of assets, so a wider universe buys a wider ball at a fixed confidence level, and a longer sample shrinks it.

`scale` states the units. A radius multiplies a norm of the weight vector, so it is in the units of the loss the ball is drawn around, and no caller can intuit that number from the confidence level alone. `scale = nothing` reads the average per-asset dispersion of the series the slot owner prices instead, which is the first pass the form's source recommends, and it moves with the sample the way the rest of the rule does.

**A drawdown owner is read on a drawdown scale.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) measures the transport cost of its own programme against the per-asset drawdown sample, so the ball it prices is a ball over drawdown scenarios and the radius carries drawdown units. `series` names that quantity, [`calibration_series_dispersion`](@ref) takes the dispersion off the drawdown sample, and `pr.sigma` reaches nothing there: it is a moment of the returns.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ScenarioCount`](@ref). The radius prices estimation error, and the error of a weighted estimate falls with Kish's effective sample size rather than with the number of rows. [`RateRadius`](@ref) reads the raw row count instead, because a rate speaks of the length of the record.

The rule carries no range check of its own. It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a radius outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> ConcentrationRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, `scale` defaults to `nothing`, which reads the average per-asset dispersion off the sample, and `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`DualNormRadius`](@ref): answers what the sampling error is in the ground metric the slot names, so its number changes with the key while this one's does not.
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct ConcentrationRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    $(field_dict[:cal_confidence])
    """
    confidence
    """
    $(field_dict[:cal_scale])
    """
    scale
    """
    $(field_dict[:cal_series_scale])
    """
    series
    function ConcentrationRadius(confidence::Number, scale::Option{<:Number},
                                 series::AbstractCalibrationSeries)
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale), typeof(series)}(confidence, scale,
                                                                      series)
    end
end
function ConcentrationRadius(; confidence::Number = 0.95, scale::Option{<:Number} = nothing,
                             series::AbstractCalibrationSeries = ReturnsSeries())
    return ConcentrationRadius(confidence, scale, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius that the concentration of measure gives on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= s \\sqrt{\\dfrac{\\chi^{2}_{N,\\, q}}{T_{e}}}\\,,\\\\
s &= \\begin{cases}
\\dfrac{1}{N} \\sum\\limits_{i=1}^{N} \\hat{s}_{i} & \\textrm{if } \\texttt{scale} \\textrm{ is } \\texttt{nothing}\\\\
\\texttt{scale} & \\textrm{otherwise}
\\end{cases}\\,,\\\\
T_{e} &= \\begin{cases}
T & \\textrm{if } w \\textrm{ is } \\texttt{nothing}\\\\
\\dfrac{\\left(\\sum\\limits_{i=1}^{T} w_{i}\\right)^{2}}{\\sum\\limits_{i=1}^{T} w_{i}^{2}} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - $(math_dict[:cal_s_radius])
  - ``\\chi^{2}_{N,\\, q}``: Quantile of the chi-squared distribution with ``N`` degrees of freedom at confidence level ``q``.
  - $(math_dict[:N])
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_w_i])

# Arguments

  - `alg`: The rule. Its `series` field names the quantity the ball is drawn around, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The radius is the same for every key, so the two tails of a Range measure that carry one rule resolve to one number.
  - `pr`: Prior result the sample size, the asset count and, on a returns series, the covariance matrix are read off. A drawdown series reads the sample instead.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ConcentrationRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    N = size(pr.X, 2)
    T = isnothing(w) ? size(pr.X, 1) : sum(w)^2 / sum(abs2, w)
    scale = if isnothing(alg.scale)
        Statistics.mean(calibration_series_dispersion(alg.series, pr))
    else
        alg.scale
    end
    q = Distributions.cquantile(Distributions.Chisq(N),
                                one(alg.confidence) - alg.confidence)
    return scale * sqrt(q / T)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius that shrinks with the square root of the sample length.

The radius is `c / sqrt(T)`. The rate is the part of the form to trust, and the coefficient is the part to calibrate: a cross-validation over `c` is the honest route to a radius, and this is the shape a grid moves over.

The rule reads the raw row count, and not the effective sample size that [`ConcentrationRadius`](@ref) reads. The rate is a statement about the length of the record, on the same terms as [`RateSignificance`](@ref).

`c` carries the units of the returns, because the rate itself is dimensionless.

The rule carries no range check of its own, on the same terms as [`ConcentrationRadius`](@ref).

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    RateRadius(;
        c::Number = 1
    ) -> RateRadius

Keywords correspond to the struct's fields. `c` defaults to `1`, which is the plain ``1/\\sqrt{T}`` rate.

## Validation

  - `c > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`DualNormRadius`](@ref): answers what the sampling error is in the ground metric the slot names, so its number changes with the key while this one's does not.
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct RateRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    $(field_dict[:cal_rate_c])
    """
    c
    function RateRadius(c::Number)
        assert_nonempty_gt0_finite_val(c, :c)
        return new{typeof(c)}(c)
    end
end
function RateRadius(; c::Number = 1)
    return RateRadius(c)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius at the square-root rate of the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= \\frac{c}{\\sqrt{T}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - ``c``: Rate coefficient, in the units of the returns.
  - $(math_dict[:T])

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The rate is the same for every key.
  - `pr`: Prior result the sample length is read off.
  - `w`: Effective observation weights. This rule reads the raw row count, so it ignores them.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`RateRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`DimensionalRateRadius`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateRadius)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    return alg.c / sqrt(size(pr.X, 1))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius that shrinks at the dimensional rate a Wasserstein ball earns, not at the square-root rate.

The radius is the Esfahani-Kuhn [drcvar](@cite) form, a scale in the units of the series the slot owner prices times the sample factor `log(1 / (1 - confidence)) / T` raised to the power `1 / max(N, 2)`. A ball of probability measures must hold the true **measure**, and not merely the true mean, and the measure-concentration result the form is read off charges that exponent for it. At `N = 2` the rule returns the square-root rate that [`ConcentrationRadius`](@ref) and [`RateRadius`](@ref) both carry, so the family's existing behaviour is the two-dimensional corner of this one.

**The ball hardly shrinks over a wide universe, and that is the model speaking rather than a defect.** At `N = 20` and `T = 250` the factor `T^(-1/20)` is `0.76`, so a record ten times longer buys almost nothing. This is the curse of dimensionality of the Wasserstein ball. A caller who reads a square-root rate and expects an estimation error reads a statement about a *mean*, and this rule makes one about a *measure*.

**The radius is conservative in practice, and its source recommends a cross-validation over the radius in its place.** A portfolio priced against a ball this wide can hold nothing but cash. [`RateRadius`](@ref) is the shape such a grid moves over, so calibrate there and read this rule as the statement of what the rate is.

The source result carries a second branch for a short record, whose exponent is `1 / a` for a tail-decay exponent `a` rather than `1 / max(N, 2)`. That branch binds only below a threshold that depends on constants the source leaves to the caller, so this rule drops it and ships the one branch. A reader who holds the paper beside this form finds the difference stated here, rather than by deriving it.

**The branch stays dropped, and [`HillTailDecay`](@ref) does not supply it.** That rule estimates the tail index of a **power law**, which is the heavy-tailed regime. The `a` of the second branch is the exponent of the **light-tail** moment condition the concentration result assumes, and the same condition carries the two constants that place the threshold. So the two exponents share a letter and nothing else, and a rule that read one for the other would state a rate the result does not give. The threshold is also unreachable in practice: it puts the short branch below a record of a few dozen rows, which no fold of a portfolio problem uses.

`N` is the asset count, `size(pr.X, 2)`, because the ball is over the assets. A factor prior carries a smaller effective dimension, and the ball is still over the assets, so a factor prior does not move `N`.

`scale` states the units. A radius multiplies a norm of the weight vector, so it is in the units of the loss the ball is drawn around, and no caller can intuit that number from the confidence level alone. `scale = nothing` reads the average per-asset dispersion of the series the slot owner prices instead, on the same terms as [`ConcentrationRadius`](@ref), and a drawdown owner is read on a drawdown scale there for the reason that rule states.

[`DistributionallyRobustConditionalDrawdownatRisk`](@ref) prices a ball around the drawdown scenarios. The scenario dimension there is still `N`, so the rate carries, and the scale moves with the series: [`bind_series`](@ref) puts the owner's marker on the rule, and [`calibration_series_dispersion`](@ref) then reads the per-asset dispersion off the drawdown sample rather than off `pr.sigma`. A drawdown column is a running functional of its returns, so its dispersion is the wider of the two, and a `scale` of `nothing` therefore gives a wider ball on that owner than on a returns owner of the same sample. A stated `scale` still wins, and it is the way to price a ball whose units are neither.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ConcentrationRadius`](@ref) and [`ScenarioCount`](@ref). The rate is a concentration statement, so the record it prices is the one Kish's count measures. [`RateRadius`](@ref) reads the raw row count instead, because its rate speaks of the length of the record.

The rule carries no range check of its own, on the same terms as [`ConcentrationRadius`](@ref). It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a radius outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DimensionalRateRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> DimensionalRateRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, `scale` defaults to `nothing`, which reads the average per-asset dispersion off the sample, and `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar])
"""
@concrete struct DimensionalRateRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    $(field_dict[:cal_dim_confidence])
    """
    confidence
    """
    $(field_dict[:cal_dim_scale])
    """
    scale
    """
    $(field_dict[:cal_series_scale])
    """
    series
    function DimensionalRateRadius(confidence::Number, scale::Option{<:Number},
                                   series::AbstractCalibrationSeries)
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale), typeof(series)}(confidence, scale,
                                                                      series)
    end
end
function DimensionalRateRadius(; confidence::Number = 0.95,
                               scale::Option{<:Number} = nothing,
                               series::AbstractCalibrationSeries = ReturnsSeries())
    return DimensionalRateRadius(confidence, scale, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius that the measure-concentration bound gives on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= s \\left(\\dfrac{\\ln\\left(\\dfrac{1}{1 - q}\\right)}{T_{e}}\\right)^{\\frac{1}{\\max(N,\\, 2)}}\\,,\\\\
s &= \\begin{cases}
\\dfrac{1}{N} \\sum\\limits_{i=1}^{N} \\hat{s}_{i} & \\textrm{if } \\texttt{scale} \\textrm{ is } \\texttt{nothing}\\\\
\\texttt{scale} & \\textrm{otherwise}
\\end{cases}\\,,\\\\
T_{e} &= \\begin{cases}
T & \\textrm{if } w \\textrm{ is } \\texttt{nothing}\\\\
\\dfrac{\\left(\\sum\\limits_{i=1}^{T} w_{i}\\right)^{2}}{\\sum\\limits_{i=1}^{T} w_{i}^{2}} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - $(math_dict[:cal_s_radius])
  - ``q``: Confidence level the bound is read at.
  - $(math_dict[:N])
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_w_i])

The exponent is floored at one half, so a universe of one or two assets returns the square-root rate rather than a faster one. The bound states no rate above that floor.

# Arguments

  - `alg`: The rule. Its `series` field names the quantity the ball is drawn around, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The radius is the same for every key, so the two tails of a Range measure that carry one rule resolve to one number.
  - `pr`: Prior result the sample size, the asset count and, on a returns series, the covariance matrix are read off. A drawdown series reads the sample instead.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`DimensionalRateRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::DimensionalRateRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    N = size(pr.X, 2)
    T = isnothing(w) ? size(pr.X, 1) : sum(w)^2 / sum(abs2, w)
    scale = if isnothing(alg.scale)
        Statistics.mean(calibration_series_dispersion(alg.series, pr))
    else
        alg.scale
    end
    return scale * (log(inv(one(alg.confidence) - alg.confidence)) / T)^inv(max(N, 2))
end
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius in the ground metric that the slot it stands in names, so that two slots of two different norms get two different numbers.

The eight radius slots of the library do not measure distance in one norm. A radius multiplies a norm of the weight vector, and the ground metric of the ball is the dual of that norm, so the `l1` coefficient of [`JuMPOptimiser`](@ref) is a distance in the ∞-norm while its `linf` coefficient is a distance in the 1-norm. This rule reads `key`, picks the ground metric of that slot, and returns the sampling error of the empirical measure in it. [`ConcentrationRadius`](@ref) and [`RateRadius`](@ref) return one number for every key, which is right inside one measure and wrong across the eight slots.

The sampling error of the mean vector is the part a linear loss sees, and its per-asset scale is the dispersion of the series the slot owner prices over the square root of the effective sample size. The radius is a norm of that error vector, at the confidence level `confidence` states.

**No term of the covariance error belongs in that vector.** A radius multiplies a norm of the weight vector, so it prices the part of the loss that is **linear** in the weights, and the loss of a distributionally robust measure is linear in the sample point. The error of the second moment reaches the objective through the tail term rather than through the ball, and the tail term carries its own coefficient, which [`TailTermParity`](@ref) sets. A radius that carried both would price one error twice.

`confidence` is a **per-coordinate** level, and it is not corrected for the number of assets. The ∞-norm case is a maximum over `N` coordinates, so a per-coordinate level understates it, and a caller who wants a level over the whole vector states the corrected number themselves, as `1 - (1 - c) / N`.

**The correction stays with the caller, and it is not a rounding.** A Bonferroni step from `0.95` to `1 - 0.05 / N` moves the quantile by about a third at four assets and by more at twenty. It is also one choice of several: a level over the vector read off `N` correlated coordinates and one read off `N` independent ones disagree by an amount only the correlation states. The source result states a per-coordinate level and no correction, so the rule ships what the result states and names the corrected number in prose. A rule that corrected silently would give a caller a level they did not ask for and cannot see.

The 1-norm case sums the per-asset errors, which prices them as if they moved together. That is the worst case over the correlations, and it is therefore the conservative reading for a radius. A correlation-aware form would give a smaller ball, and this rule does not compute one.

**The worst case is the reading a radius wants, so the correlation-aware form stays out.** A radius that understates the ball gives a model that is robust to less than the caller asked for, and the failure is silent: the optimisation solves and the weights are not robust. A radius that overstates it is visible, because the portfolio moves towards cash. The two errors are not symmetric, so the rule takes the one a caller can see.

`p` serves the `:lpreg_val` slot alone. The ground metric of [`LpRegularisation`](@ref) is the type-``q`` metric with ``1/p + 1/q = 1``, and `key` names the slot rather than the norm order. The order belongs to the penalty, so that site fills this field through [`bind_norm_order`](@ref) before it resolves the slot, and the call **overwrites** whatever the field holds. A stated `p` therefore serves a caller who runs the rule outside that site, and nothing else. Every other key ignores the field.

**The drawdown owner is served on a drawdown scale, and `series` is what says so.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) measures the transport cost of its own programme against the per-asset drawdown sample, so the ball it prices is a ball over drawdown scenarios and the ground metric is a distance between two such vectors. [`calibration_series_dispersion`](@ref) reads the error scale off that sample under a drawdown marker, and `pr.sigma` reaches nothing there: it is a moment of the returns, and no scaling of it states a moment of a drawdown. The ground metric does not move with the series, only the vector it is taken of.

**The error scale of a drawdown series is a floor, and the record is the reason.** A drawdown is a running functional, so its entries are strongly dependent down a column, and ``\\hat{s}_{i} / \\sqrt{T_{e}}`` prices a record of ``T_{e}`` independent draws that a drawdown sample does not hold. The rule states no correction for it, on the same terms it states none for the number of assets: a correction needs a model of the dependence, and the sample states none. So the reading is the honest scale of the quantity, taken as if the record were independent.

`T_e` is Kish's effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ConcentrationRadius`](@ref).

The rule carries no range check of its own, on the same terms as [`ConcentrationRadius`](@ref).

**A sample of no dispersion gives a radius of zero, and the slot admits it.** Every diagonal of the dispersion is non-negative, so the radius is non-negative and finite for every admissible input. A sample whose columns never move gives an error vector of zeros, and a ball of no width is the right answer for it: the empirical measure is the only measure such a sample supports, and the robust measure reduces to the plain one it is drawn around. In floating point that sample gives a number many orders below the smallest one a caller would state rather than an exact zero, so the slot owner's `> 0` check passes and the model prices a ball of no width. Both outcomes read the same way, and neither is a defect.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DualNormRadius(;
        confidence::Number = 0.95,
        p::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> DualNormRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, `p` defaults to `nothing`, which serves every slot but `:lpreg_val`, and `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `0 < confidence < 1`.
  - If `p` is not `nothing`: `isfinite(p)` and `p > 1`, on the same terms as [`LpRegularisation`](@ref).

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`ConcentrationRadius`](@ref): answers how wide the ball is at a confidence level, in one dimensionless factor that no norm enters.
  - [`RateRadius`](@ref): answers how fast the ball shrinks with the record, and leaves the coefficient to a cross-validation.
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`bind_series`](@ref)
  - [`bind_norm_order`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct DualNormRadius <: AbstractAmbiguityRadiusCalibrationAlgorithm
    """
    `confidence`: Per-coordinate confidence level of the normal quantile the error scale is multiplied by. It is not corrected for the number of assets, so the ∞-norm case reads a level over one coordinate and not over the vector.
    """
    confidence
    """
    `p`: Norm order of the [`LpRegularisation`](@ref) penalty the radius stands in, or `nothing`. It is read for the `:lpreg_val` key alone, where the ground metric is the type-``q`` metric with ``1/p + 1/q = 1`` and no key can name ``q``. The penalty site overwrites it through [`bind_norm_order`](@ref), so state it only to run the rule outside that site.
    """
    p
    """
    $(field_dict[:cal_series_scale])
    """
    series
    function DualNormRadius(confidence::Number, p::Option{<:Number},
                            series::AbstractCalibrationSeries)
        assert_unit_interval(confidence, :confidence)
        if !isnothing(p)
            @argcheck(isfinite(p), IsNonFiniteError)
            @argcheck(p > one(p), DomainError)
        end
        return new{typeof(confidence), typeof(p), typeof(series)}(confidence, p, series)
    end
end
function DualNormRadius(; confidence::Number = 0.95, p::Option{<:Number} = nothing,
                        series::AbstractCalibrationSeries = ReturnsSeries())
    return DualNormRadius(confidence, p, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius of the slot `key` names, in the ground metric of that slot, on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= z_{c} \\lVert \\boldsymbol{e} \\rVert_{g}\\,,\\\\
e_{i} &= \\dfrac{\\hat{s}_{i}}{\\sqrt{T_{e}}}\\,,\\\\
T_{e} &= \\begin{cases}
T & \\textrm{if } w \\textrm{ is } \\texttt{nothing}\\\\
\\dfrac{\\left(\\sum\\limits_{i=1}^{T} w_{i}\\right)^{2}}{\\sum\\limits_{i=1}^{T} w_{i}^{2}} & \\textrm{otherwise}
\\end{cases}\\,,\\\\
q &= \\dfrac{p}{p - 1}\\,.
\\end{align}
```

Where:

  - $(math_dict[:cal_r_radius])
  - ``z_{c}``: Quantile of the standard normal distribution at the per-coordinate confidence level ``c``.
  - ``\\boldsymbol{e}``: Per-asset scale of the sampling error of the mean vector, in the units of the series the slot owner prices.
  - ``g``: Order of the ground metric, which `key` names.
  - $(math_dict[:cal_s_i_series])
  - $(math_dict[:T])
  - $(math_dict[:cal_T_e])
  - $(math_dict[:cal_w_i])
  - ``p``: Norm order of the penalty, the `p` field.
  - ``q``: Order of the type-``q`` ground metric of that penalty.

# The ground metric of each key

| `key`          | Slot                                                                                                              | Penalised norm | Ground metric ``g`` |
|:-------------- |:----------------------------------------------------------------------------------------------------------------- |:-------------- |:------------------- |
| `:l1`          | `l1` of [`JuMPOptimiser`](@ref)                                                                                   | 1              | ``\\infty``         |
| `:linf`        | `linf` of [`JuMPOptimiser`](@ref)                                                                                 | ``\\infty``    | 1                   |
| `:r`           | [`DistributionallyRobustConditionalValueatRisk`](@ref), [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) | ``\\infty``    | 1                   |
| `:r_a`, `:r_b` | [`DistributionallyRobustConditionalValueatRiskRange`](@ref)                                                       | ``\\infty``    | 1                   |
| `:l2reg_val`   | `val` of [`L2Regularisation`](@ref)                                                                               | 2              | 2                   |
| `:lpreg_val`   | `val` of [`LpRegularisation`](@ref)                                                                               | ``p``          | ``q``               |

The two ends of a Range measure carry one ground metric, so a rule stated on both resolves to one number. A radius names no end of the distribution.

# Arguments

  - `alg`: The rule. Its `series` field names the quantity the ball is drawn around, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. It selects the ground metric, so this is the one rule of its family for which the key carries meaning.
  - `pr`: Prior result the sample size and, on a returns series, the covariance matrix are read off. A drawdown series reads the sample instead.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Validation

  - `key` is one of the seven the table names, else an `ArgumentError` naming the key it received and the keys it serves.
  - The `:lpreg_val` key needs a stated `p`, else an `ArgumentError` naming the field.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`DualNormRadius`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_dispersion`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::DualNormRadius)(key::Symbol, pr::AbstractPriorResult, w, ::Any)
    T = isnothing(w) ? size(pr.X, 1) : sum(w)^2 / sum(abs2, w)
    e = calibration_series_dispersion(alg.series, pr) ./ sqrt(T)
    z = Distributions.quantile(Distributions.Normal(), alg.confidence)
    return z * dual_norm_radius_scale(alg, key, e)
end
"""
    dual_norm_radius_scale(alg::DualNormRadius, key::Symbol, e::AbstractVector)

Reduce the per-asset error vector `e` to one number, in the ground metric that `key` names.

This is the whole of the key's meaning, held apart from the functor. Six of the seven keys name a fixed norm order, so they are a literal table and not a chain of branches. The seventh, `:lpreg_val`, derives its order from the `p` the rule carries, because no key can name it. An eighth key is a refusal, because a caller who writes their own measure reaches it first and the message must name the keys the rule serves.

# Arguments

  - `alg`: The rule, read for `p` on the `:lpreg_val` arm alone.
  - `key`: Name of the slot that is being resolved.
  - `e`: Per-asset scale of the sampling error of the mean vector.

# Validation

  - `key` is one of the seven [`DualNormRadius`](@ref) serves, else an `ArgumentError`.
  - `alg.p` is stated when `key` is `:lpreg_val`, else an `ArgumentError`.

# Returns

  - `s::Number`: The norm of `e` in the ground metric of the slot.

# Related

  - [`DualNormRadius`](@ref)
  - [`LpRegularisation`](@ref)
"""
function dual_norm_radius_scale(alg::DualNormRadius, key::Symbol, e::AbstractVector)
    metrics = (; l1 = Inf, linf = 1, r = 1, r_a = 1, r_b = 1, l2reg_val = 2)
    g = if key === :lpreg_val
        @argcheck(!isnothing(alg.p),
                  ArgumentError("`DualNormRadius.p` is `nothing` while the `:lpreg_val` slot is being resolved. That slot's ground metric is the type-`q` metric of an `LpRegularisation` penalty, with `1/p + 1/q = 1`, and the order belongs to the penalty. The penalty site fills the field through `bind_norm_order`, so a `nothing` here means the rule was resolved somewhere that binds no order. Place the rule in the `val` field of an `LpRegularisation`, or state `p` on the rule."))
        alg.p / (alg.p - one(alg.p))
    else
        get(() -> throw(ArgumentError("`DualNormRadius` reads `key` to pick the ground metric of the slot it stands in, and it received `:$key`, which names no slot it serves. The keys it serves are `:l1`, `:linf`, `:r`, `:r_a`, `:r_b`, `:l2reg_val` and `:lpreg_val`. A measure of your own that holds a radius resolves its slot under one of those keys, or carries a rule of its own.")),
            metrics, key)
    end
    return LinearAlgebra.norm(e, g)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes the Esfahani-Kuhn tail weight that prices the tail term of the loss at a stated multiple of its mean term.

At ``r \\to 0`` the loss of [`DistributionallyRobustConditionalValueatRisk`](@ref) reduces to ``-\\mathbb{E}[\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}] + l \\, \\mathrm{CVaR}_{\\alpha}(\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi})``, so `l` is the exchange rate between one unit of expected return and one unit of tail loss. Both terms carry the units of the returns, so `l` is dimensionless. **A dimensionless number is not a scale-free one, and that is the trap this rule exists for.** The mean of a daily return column sits two orders of magnitude below its five per cent CVaR, and on a monthly column the gap is far smaller, so a stated `l = 1.0` prices the tail term about forty times the mean term on a daily sample and under ten times on a monthly one. One stated number is a different trade-off at every sampling frequency, and at every re-fit that changes the record.

The rule carries the sample's own units and nothing else, so the preference stays the caller's. `ratio` states how many mean terms one tail term is worth, and the rule returns the `l` that prices it so on the sample the prior result carries. `ratio = 1` is parity, and `ratio = 2` prices the tail term at twice the mean term on every sample.

A rule reads no portfolio, so it cannot form ``\\boldsymbol{w}^{\\intercal} \\boldsymbol{\\xi}``. The series it reads is the cross-section of the per-asset losses of the quantity the slot owner prices: the mean-term scale ``m`` is the mean of that pool, and the tail-term scale ``c`` is the mean of the per-column ``\\mathrm{CVaR}_{\\alpha}``. **The mean of the per-column values is not the pooled ``\\mathrm{CVaR}_{\\alpha}``, and the difference is deliberate.** A pooled tail is drawn from the worst columns, so it is dominated by the most volatile assets and gives a smaller weight. The measure prices a portfolio rather than a single asset, so the per-column mean is the reading that answers it.

**A caller who holds a reference portfolio still cannot state one, and that is the design.** The pooled cross-section is the equal-weight reading, and a stated reference portfolio would give a sharper one. It would also make the rule a second place where a portfolio enters the problem, and the calibration channel carries no portfolio by construction: [`resolve_calibration_slot`](@ref) hands a rule the Prior and nothing else, so a reference vector would have to travel a channel of its own, be checked against the universe of the fold, and be re-stated whenever a subset view or a cluster changed that universe. The exchange rate the rule prices is a property of the **sample**, and the equal-weight cross-section is the reading of it that needs no second input. A caller who wants the rate at their own portfolio computes `l` themselves and states the number.

``m`` is negative for a sample of positive expected return, and the rule takes ``\\lvert m \\rvert``. No field states the sign: a negative weight is not admissible in the slot, and a sample of negative expected return does not turn the trade-off around.

The rule reads its sibling `alpha`, because ``c`` is a ``\\mathrm{CVaR}`` at the measure's own significance level. `alpha` and `l` are a **travelling pair**, on the same terms as `alpha` and `kappa`: [`bind_alpha`](@ref) fills the `alpha` field with the number the slot owner resolved, and the owner's own resolution method resolves `alpha` first for that reason. A caller who runs the rule outside a measure states `alpha` instead.

**Both scales read the observation weights.** They are sample statistics rather than counts, so a weighted sample is read weighted. [`RateRadius`](@ref) and [`RateSignificance`](@ref) ignore `w` because a rate speaks of the length of the record, and that reading does not carry to a moment.

**`series` says which quantity both terms are read over, and the slot owner states it.** [`DistributionallyRobustConditionalDrawdownatRisk`](@ref) carries the key `:l` as well, and its tail term is a ``\\mathrm{CDaR}`` of the portfolio drawdown series. A rule is given no portfolio, but it can form the drawdown series of each **column**, and [`bind_series`](@ref) puts the marker there at the resolution site. Both terms then move together: the mean term is the mean drawdown of the pool, and the tail term is the mean of the per-column ``\\mathrm{CDaR}_{\\alpha}``. The same [`ConditionalValueatRisk`](@ref) reading forms it, because the tail mean of a non-positive drawdown column **is** the ``\\mathrm{CDaR}`` of that column, so the rule and the measure it calibrates still cannot drift apart.

A second reading of `l` exists, and this rule does not take it. `l` can be read as a risk-aversion coefficient and mapped from a mean-variance one, but a variance penalty is quadratic in the weight vector and a ``\\mathrm{CVaR}`` term is positively homogeneous, so the two objectives are not comparable term by term. The map holds at one reference portfolio and nowhere else, and a rule gets no portfolio.

The rule carries no range check on the weight it returns, on the same terms as [`RateRadius`](@ref). The slot owner's constructor keeps that job. Its two refusals state that the ratio exists at all, which is the shape of the one check [`EntropyBudget`](@ref) carries. A sample whose mean loss is near zero returns a very large weight, and that is the sample speaking rather than a defect: no threshold separates a small mean loss from a smaller one.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    TailTermParity(;
        ratio::Number = 1,
        alpha::Option{<:Number} = nothing,
        series::AbstractCalibrationSeries = ReturnsSeries()
    ) -> TailTermParity

Keywords correspond to the struct's fields. `ratio` defaults to `1`, which is parity between the two terms. `alpha` defaults to `nothing`, which is the state a rule stands in a slot in. `series` defaults to [`ReturnsSeries`](@ref), and every slot owner overwrites it through [`bind_series`](@ref).

## Validation

  - `ratio > 0` and finite.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`bind_alpha`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`RateRadius`](@ref)
  - [`resolve_calibration_slot`](@ref)

# References

  - $(ref_dict[:drcvar])
"""
@concrete struct TailTermParity <: AbstractAmbiguityTailWeightCalibrationAlgorithm
    """
    $(field_dict[:cal_ratio])
    """
    ratio
    """
    $(field_dict[:cal_alpha_sib])
    """
    alpha
    """
    $(field_dict[:cal_series_twt])
    """
    series
    function TailTermParity(ratio::Number, alpha::Option{<:Number},
                            series::AbstractCalibrationSeries)
        assert_nonempty_gt0_finite_val(ratio, :ratio)
        return new{typeof(ratio), typeof(alpha), typeof(series)}(ratio, alpha, series)
    end
end
function TailTermParity(; ratio::Number = 1, alpha::Option{<:Number} = nothing,
                        series::AbstractCalibrationSeries = ReturnsSeries())
    return TailTermParity(ratio, alpha, series)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the tail weight that prices the tail term at `ratio` mean terms on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
l &= \\rho \\dfrac{\\lvert m \\rvert}{c}\\,,\\\\
y_{tj} &= \\left(\\mathcal{S}\\left(\\boldsymbol{r}_{j}\\right)\\right)_{t}\\,,\\\\
m &= -\\dfrac{\\sum\\limits_{t=1}^{T} \\sum\\limits_{j=1}^{N} w_{t} y_{tj}}{N \\sum\\limits_{t=1}^{T} w_{t}}\\,,\\\\
c &= \\dfrac{1}{N} \\sum\\limits_{j=1}^{N} \\mathrm{CVaR}_{\\alpha}\\left(\\boldsymbol{y}_{j}\\right)\\,.
\\end{align}
```

Where:

  - ``l``: Esfahani-Kuhn tail weight, the weight of the tail term of the loss.
  - ``\\rho``: Number of mean terms that one tail term is worth.
  - ``m``: Mean-term scale, the mean of the pooled cross-section of the per-asset losses.
  - ``c``: Tail-term scale, the mean of the per-column ``\\mathrm{CVaR}_{\\alpha}`` of the loss.
  - ``\\boldsymbol{r}_{j}``: Column ``j`` of the returns matrix.
  - ``\\mathcal{S}``: The series `alg.series` names, built from one column. It is the identity on a [`ReturnsSeries`](@ref), and a drawdown series on the two markers of [`AbstractDrawdownSeries`](@ref).
  - ``y_{tj}``: Entry ``t`` of the series of column ``j``.
  - ``\\boldsymbol{y}_{j}``: The series of column ``j``. A ``\\mathrm{CVaR}_{\\alpha}`` of a non-positive drawdown series is the ``\\mathrm{CDaR}_{\\alpha}`` of that column.
  - ``w_{t}``: Observation weight of period ``t``. Every weight is one when none is stated.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - $(math_dict[:N])
  - $(math_dict[:r_tj])

Every column holds ``T`` entries, so the pooled mean and the mean of the per-column means are one number, and the pooled form is the one written.

# Algorithm

 1. Build the sample `alg.series` names with [`calibration_series_matrix`](@ref) into `X`, and read the effective observation weights off `w`. A [`ReturnsSeries`](@ref) returns `pr.X` itself.
 2. Form the mean-term scale `m`, the negated weighted mean of the pooled cross-section of `X`.
 3. Build a [`ConditionalValueatRisk`](@ref) at `alg.alpha` carrying the same weights, and form the tail-term scale `c`, the mean over the columns of `X` of the value that measure takes on each. The measure's own reading is the one used, so the rule and the measure it calibrates cannot drift apart.
 4. Return `alg.ratio * abs(m) / c`.

# Arguments

  - `alg`: The rule. Its `alpha` field must hold a number, which [`bind_alpha`](@ref) puts there, and its `series` field names the quantity, which [`bind_series`](@ref) puts there.
  - `key`: Name of the slot that is being resolved. The scales are read off the asset columns, so the key never selects the value, and the two ends of a Range measure part company through their two probabilities alone.
  - `pr`: Prior result the returns matrix is read off. The series is built from its columns.
  - `w`: Effective observation weights, or `nothing`. Both scales are sample statistics, so this rule reads them.
  - `slv`: Effective solver. This rule needs none, because both scales are closed forms.

# Validation

  - `alg.alpha` must not be `nothing`.
  - ``\\lvert m \\rvert`` must be positive.
  - ``c`` must be positive.

# Returns

  - `l::Number`: The Esfahani-Kuhn tail weight.

# Related

  - [`TailTermParity`](@ref)
  - [`bind_alpha`](@ref)
  - [`bind_series`](@ref)
  - [`calibration_series_matrix`](@ref)
  - [`ConditionalValueatRisk`](@ref)
  - [`ConditionalDrawdownatRisk`](@ref)
  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::TailTermParity)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    @argcheck(!isnothing(alg.alpha),
              IsNothingError("`TailTermParity.alpha` is `nothing`, so the rule cannot form the tail-term scale `c`, which is a CVaR at the measure's own significance level. The probability travels to the rule through `bind_alpha`, which the slot owner calls after it resolves `alpha`. State `alpha` on the rule itself to run it outside a measure."))
    # Both scales are read over the series the OWNER prices. Under a returns marker this is
    # `pr.X` itself, and under a drawdown marker the tail term is a CDaR: the CVaR kernel
    # over a non-positive drawdown column is the mean of its worst `alpha`.
    X = calibration_series_matrix(alg.series, pr.X)
    N = size(X, 2)
    ws = get_observation_weights(w, view(X, :, 1))
    m = if isnothing(ws)
        -Statistics.mean(X)
    else
        -sum(j -> LinearAlgebra.dot(view(X, :, j), ws), axes(X, 2)) / (N * sum(ws))
    end
    @argcheck(abs(m) > 0,
              DomainError(m,
                          "`TailTermParity` read a mean-term scale of $m on the sample, so the ratio `ratio * abs(m) / c` is zero and the slot admits no such weight. The rule prices one term of the loss against the other, and a sample whose pooled mean loss is zero states no exchange rate between them: every tail weight prices a mean term of zero alike. State `l` on the slot instead."))
    # The tail term the weight scales is a CVaR, so the reading is the measure's own rather
    # than a second encoding of it here. A rule that carried its own copy would drift from
    # the measure it calibrates the moment either moved.
    rm = ConditionalValueatRisk(; alpha = alg.alpha, w = w)
    c = Statistics.mean(j -> rm(view(X, :, j)), axes(X, 2))
    @argcheck(c > 0,
              DomainError(c,
                          "`TailTermParity` read a tail-term scale of $c at `alpha = $(alg.alpha)`, which is not positive, so the ratio `ratio * abs(m) / c` has no admissible value. The scale is the mean of the per-column CVaR of the loss, and a non-positive one is a sample whose worst `alpha` of every column holds no loss at all. Widen `alpha`, or state `l` on the slot."))
    return alg.ratio * abs(m) / c
end
# The other methods of `bind_alpha` stand beside the deformation rules, which are the pair
# the verb was built for. These two stand here because the types they name are declared
# above them, and the role is rebuilt around the bound rule on the same terms.
function bind_alpha(r::AmbiguityTailWeightCalibration, alpha::Number)
    return AmbiguityTailWeightCalibration(; alg = bind_alpha(r.alg, alpha))
end
function bind_alpha(alg::TailTermParity, alpha::Number)
    return TailTermParity(; ratio = alg.ratio, alpha = alpha, series = alg.series)
end
# The same holds for the six `bind_series` methods below. The two ambiguity families are
# declared between the verb and this point, so their methods cannot stand beside the
# deformation pair the verb was written for.
function bind_series(r::AmbiguityRadiusCalibration, series::AbstractCalibrationSeries)
    return AmbiguityRadiusCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(r::AmbiguityTailWeightCalibration, series::AbstractCalibrationSeries)
    return AmbiguityTailWeightCalibration(; alg = bind_series(r.alg, series))
end
function bind_series(alg::ConcentrationRadius, series::AbstractCalibrationSeries)
    return ConcentrationRadius(; confidence = alg.confidence, scale = alg.scale,
                               series = series)
end
function bind_series(alg::DimensionalRateRadius, series::AbstractCalibrationSeries)
    return DimensionalRateRadius(; confidence = alg.confidence, scale = alg.scale,
                                 series = series)
end
function bind_series(alg::DualNormRadius, series::AbstractCalibrationSeries)
    return DualNormRadius(; confidence = alg.confidence, p = alg.p, series = series)
end
function bind_series(alg::TailTermParity, series::AbstractCalibrationSeries)
    return TailTermParity(; ratio = alg.ratio, alpha = alg.alpha, series = series)
end
"""
$(DocStringExtensions.TYPEDEF)

Computes a norm ceiling that holds a stated fraction of the universe effective, so that the floor refits whenever the universe changes.

A norm ceiling and the effective number of assets are reciprocally related, so a bound on the norm is a floor on that count. This rule states the floor as a **fraction of the universe** rather than as a count. The asset count comes off the prior result, so a subset view, a cluster and a cross-validation fold each get the floor their own universe earns, and no number is pinned to the universe it was written for.

`p` is the norm order the ceiling is read against, and it belongs to the constraint. Each of the three constraint sites calls [`bind_norm_order`](@ref) before it resolves the slot, and that call **overwrites** whatever this field holds. So a stated `p` serves a caller who runs the rule outside those sites, and nothing else.

The rule carries no range check on its answer, on the same terms as [`ConcentrationRadius`](@ref). It returns the quantity of the slot it stands in, so the slot owner is the whole validation.

# Mathematical definition

```math
\\begin{align}
m &= f N\\,,\\\\
\\mathrm{val} &= \\begin{cases}
m^{-1} & \\textrm{if } p \\textrm{ is infinite}\\\\
m^{1/p - 1} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{val}``: Norm ceiling.
  - ``m``: Effective number of assets the ceiling holds as a floor.
  - ``f``: Fraction of the universe, the `fraction` field.
  - $(math_dict[:N])
  - ``p``: Norm order of the constraint the ceiling stands in.

The order-``p`` effective number of assets is ``N_{\\mathrm{eff},\\, p}(\\boldsymbol{w}) = \\left(\\sum_{i} \\lvert w_{i} \\rvert^{p}\\right)^{1/(1 - p)}``, so ``N_{\\mathrm{eff},\\, p} \\geq m`` is ``\\lVert \\boldsymbol{w} \\rVert_{p} \\leq m^{1/p - 1}``. It is the reading [`number_effective_assets`](@ref) states, taken to an arbitrary order: at ``p = 2`` the two are the same number, and at every order an equal-weight portfolio over ``m`` assets reports exactly ``m``. The exponent is also ``-1/q`` for the conjugate order ``q``, because ``1/p - 1 = -1/q``.

The infinite arm is the limit of the finite one and not a second reading. ``m^{1/p - 1} \\to m^{-1}`` as ``p`` grows, and a cap of ``1/m`` on the largest weight of a fully invested portfolio spreads it over at least ``m`` assets. So the two arms meet, and a caller who raises ``p`` towards the infinite order sees the ceiling move towards ``1/m`` rather than away from it.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    EffectiveAssetFloor(;
        fraction::Number = 0.5,
        p::Option{<:Number} = nothing
    ) -> EffectiveAssetFloor

Keywords correspond to the struct's fields. `fraction` defaults to `0.5`, which holds half of the universe effective, and `p` defaults to `nothing`, which every constraint site fills.

## Validation

  - `0 < fraction <= 1`.
  - If `p` is not `nothing`: `p >= 1`.

# Related

  - [`AbstractNormCeilingCalibrationAlgorithm`](@ref)
  - [`NormCeilingCalibration`](@ref)
  - [`bind_norm_order`](@ref)
  - [`number_effective_assets`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
@concrete struct EffectiveAssetFloor <: AbstractNormCeilingCalibrationAlgorithm
    """
    $(field_dict[:cal_fraction])
    """
    fraction
    """
    $(field_dict[:cal_norm_order])
    """
    p
    function EffectiveAssetFloor(fraction::Number, p::Option{<:Number})
        assert_nonempty_gt0_finite_val(fraction, :fraction)
        @argcheck(fraction <= one(fraction), DomainError)
        if !isnothing(p)
            @argcheck(p >= one(p), DomainError)
        end
        return new{typeof(fraction), typeof(p)}(fraction, p)
    end
end
function EffectiveAssetFloor(; fraction::Number = 0.5, p::Option{<:Number} = nothing)
    return EffectiveAssetFloor(fraction, p)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the norm ceiling that holds `fraction` of the universe that `pr` carries effective.

The universe is the column count of the prior result's returns matrix, so the ceiling moves with the universe the model actually sees.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. It names the site in the refusal below, and the ceiling itself does not depend on it: the norm order carries that dependence instead.
  - `pr`: Prior result the asset count is read off.
  - `w`: Effective observation weights. A universe count is not a sample count, so this rule ignores them.
  - `slv`: Effective solver. This rule needs none.

# Validation

  - `alg.p` is not `nothing`. Every constraint site fills it through [`bind_norm_order`](@ref), so a `nothing` here means the rule was resolved somewhere that binds no order.

# Returns

  - `val::Number`: The norm ceiling.

# Related

  - [`EffectiveAssetFloor`](@ref)
  - [`bind_norm_order`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::EffectiveAssetFloor)(key::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    p = alg.p
    @argcheck(!isnothing(p),
              ArgumentError("`$(nameof(EffectiveAssetFloor)).p` is `nothing` while the rule in `$key` is being resolved. A ceiling is read against one norm order, the order belongs to the constraint, and each constraint site fills it through `bind_norm_order`. Place the rule in `l2c`, `lpc` or `linfc`, or state `p` on the rule."))
    m = alg.fraction * size(pr.X, 2)
    return isinf(p) ? inv(m) : m^(inv(p) - one(p))
end
"""
    bind_norm_order(slot, p::Number)

Hand the norm order of a weight-norm constraint to the rule that computes its ceiling.

A **Norm Ceiling** is read against one norm order, and that order is a property of the constraint rather than of the rule: one rule placed in `lpc` serves every term, and each term carries its own `p`. [`resolve_calibration_slot`](@ref) carries a `Symbol` and no number, so the order travels through the rule itself. This is the shape [`bind_alpha`](@ref) already uses to carry a significance level to a deformation rule.

An **Ambiguity Radius** on the `val` field of [`LpRegularisation`](@ref) reads the same order, and reads it for the same reason. The ground metric of that penalty is the type-``q`` metric with ``1/p + 1/q = 1``, so [`DualNormRadius`](@ref) needs the owner's `p`, and `key` names the slot rather than the norm order. So that site binds too, and the radius family carries the same pair of methods the ceiling family carries.

The default is the identity, so a stated number crosses unchanged, and so does a caller's own plain function. A plain function reads the slot's name from `key` instead.

The order the constraint site holds **wins**. A rule that already carries one has it replaced, because the constraint is the thing the quantity is read against and the rule cannot know which site it reached.

# Arguments

  - `slot`: The slot's occupant: a number, a [`NormCeilingCalibration`](@ref), or an [`AmbiguityRadiusCalibration`](@ref).
  - `p`: Norm order of the constraint or penalty the quantity stands in.

# Returns

  - `y`: The occupant, with the order filled wherever it holds a rule that reads one.

# Related

  - [`NormCeilingCalibration`](@ref)
  - [`EffectiveAssetFloor`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`DualNormRadius`](@ref)
  - [`bind_alpha`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function bind_norm_order(slot, ::Number)
    return slot
end
function bind_norm_order(r::NormCeilingCalibration, p::Number)
    return NormCeilingCalibration(; alg = bind_norm_order(r.alg, p))
end
function bind_norm_order(alg::EffectiveAssetFloor, p::Number)
    return EffectiveAssetFloor(; fraction = alg.fraction, p = p)
end
function bind_norm_order(r::AmbiguityRadiusCalibration, p::Number)
    return AmbiguityRadiusCalibration(; alg = bind_norm_order(r.alg, p))
end
function bind_norm_order(alg::DualNormRadius, p::Number)
    return DualNormRadius(; confidence = alg.confidence, p = p, series = alg.series)
end
"""
    sigma_chol_selector(sigma, chol, pr::AbstractPriorResult)

Apply the prior fallback to a covariance slot and its factorisation **as a pair**, so that the two never come from two different sources.

`chol` is a factorisation of `sigma`. Pairing a factor with a covariance matrix it does not factorise would let the model optimise one quantity while the functor evaluates another. So the pair falls back to the prior only when the measure names neither; a stated `sigma` with no factor keeps no factor, and the consumer derives the right one.

The Deferred-Quantity state never reaches here: [`resolve_deferred_quantities`](@ref) has already turned it into the pair the fit produced. Nor does a stated `chol` without a matrix `sigma` beside it — [`assert_derived_slot_has_source`](@ref) refuses both that and a `chol` stated beside a deferred `sigma` at construction, so the two arms below see only value states.

A stated `chol` is never rebuilt from `sigma`. Under a factor prior the factorisation is sparse and special, and a rebuild would throw that structure away.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`resolve_deferred_quantities`](@ref)
  - [`chol_sigma_selector`](@ref)
"""
function sigma_chol_selector(sigma, chol, ::AbstractPriorResult)
    return sigma, chol
end
function sigma_chol_selector(::Nothing, ::Nothing, pr::AbstractPriorResult)
    return pr.sigma, pr.chol
end
"""
    assert_derived_slot_has_source(derived, source, dname::Symbol, sname::Symbol)

Refuse a derived slot that was stated without a source the caller can see.

A derived slot is a function of its source, so the two are one pair out of one fit. Two states break that, and both are refused at construction:

  - **The source is unstated.** The prior supplies it, so the caller's derived value would pair with a source the caller never saw.
  - **The source holds a Deferred Quantity.** That fit supplies the pair, so the caller's derived value would be discarded, or worse, kept beside a source it does not describe.

`chol` is a factorisation of `sigma`, and `V` is the negative spectral part of `sk`. Both follow this rule, so a stated derived slot always means a stated source value.

# Arguments

  - `derived`: The derived slot's occupant.
  - `source`: The source slot's occupant.
  - `dname`: Name of the derived slot, as the message prints it.
  - `sname`: Name of the source slot, as the message prints it.

# Validation

  - Throws an `ArgumentError` when `source` holds a [`DeferredQuantity`](@ref) and `derived` is not `nothing`. That fit supplies the pair.
  - Throws an `ArgumentError` when `derived` is not `nothing` and `source` is `nothing`. The prior would supply the source, and the caller's derived value would pair with a source the caller never saw.

# Returns

  - `nothing`.

# Related

  - [`Variance`](@ref)
  - [`StandardDeviation`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`NegativeSkewness`](@ref)
  - [`DeferredQuantity`](@ref)
"""
function assert_derived_slot_has_source(derived, source, dname::Symbol, sname::Symbol)
    if isa(source, DeferredQuantity)
        @argcheck(isnothing(derived),
                  ArgumentError("`$dname` is derived from `$sname`, so it cannot be given when `$sname` holds a Deferred Quantity. That fit supplies the pair, and a stated `$dname` would describe a `$sname` the caller never saw. Give `$sname` alone, or state both as values."))
    else
        @argcheck(isnothing(derived) || !isnothing(source),
                  ArgumentError("`$dname` is derived from `$sname`, so it cannot be given on its own. Give `$sname` as well, or give neither and let the prior supply the pair."))
    end
    return nothing
end
"""
    sel(risk_variable, source_variable)

Unified risk-measure selector emitted by the [`@pprop`](@ref)/[`@cprop`](@ref) tags. Prefers
the risk-measure value `risk_variable` when present, otherwise falls back to `source_variable`
(a prior moment for `@pprop`, or a threaded optimiser value for `@cprop`). Dispatches on
operand types to the appropriate leaf selector and inlines to zero cost:

  - solvers (`Slv_VecSlv`) → [`solver_selector`](@ref)
  - uncertainty sets (`UcSE_UcS`) → [`ucs_selector`](@ref)
  - a **Deferred Quantity** or a **Calibration Rule** → kept, because a slot the caller filled with the method that computes the value is a stated slot
  - everything else (moments) → [`nothing_scalar_array_selector`](@ref)

The Deferred-Quantity arm exists because [`@propagatable`](@ref) runs the selection **before** [`resolve_deferred_quantities`](@ref), so a `@pprop` slot that admits a Deferred Quantity reaches here still holding one. The prior must not fill such a slot: the caller stated the method, and the resolution that follows replaces it with the value that method produced. The same reading covers a Calibration Rule.

Note: the `solver_selector` both-`nothing` "cannot solve" error is not reachable through
`sel` (both-`nothing` routes to the moment selector and returns `nothing`); the
`JuMPOptimiser` solver-required invariant makes that case unreachable in the pipeline. See
ADR 0012.

# Related

  - [`@pprop`](@ref)
  - [`@cprop`](@ref)
  - [`nothing_scalar_array_selector`](@ref)
  - [`solver_selector`](@ref)
  - [`ucs_selector`](@ref)
"""
function sel(risk_variable, source_variable)
    return nothing_scalar_array_selector(risk_variable, source_variable)
end
function sel(risk_variable::Slv_VecSlv, source_variable)
    return solver_selector(risk_variable, source_variable)
end
sel(::Nothing, source_variable::Slv_VecSlv) = solver_selector(nothing, source_variable)
sel(risk_variable::UcSE_UcS, source_variable) = ucs_selector(risk_variable, source_variable)
sel(::Nothing, source_variable::UcSE_UcS) = ucs_selector(nothing, source_variable)
sel(risk_variable::DeferredQuantity, ::Any) = risk_variable
sel(risk_variable::AbstractCalibrationEstimator, ::Any) = risk_variable
"""
    _ctx(args...)

Locate the lone threaded optimiser context value (a solver, `Slv_VecSlv`) in the variadic
tail of a prior `factory` call, returning `nothing` if none is present. Emitted by the
[`@cprop`](@ref) tag as the source argument to [`sel`](@ref). The tuple scan is unrolled by
the compiler, so it is type-stable and allocation-free. See ADR 0012.

# Related

  - [`@cprop`](@ref)
  - [`sel`](@ref)
"""
_ctx() = nothing
_ctx(a::Slv_VecSlv, args...) = a
_ctx(::Any, args...) = _ctx(args...)
"""
    expected_risk(r, args...; kwargs...)

Compute the expected value of a risk measure.

Generic function extended by concrete risk measure types. Each method computes the risk value associated with its risk measure type, given a portfolio (or its return distribution).

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
"""
function expected_risk end
"""
    expected_risk_from_returns(r, X; kwargs...)

Compute the expected risk of a measure from a precomputed net-return series.

Generic function extended by concrete risk measure types that support the precomputed-returns contract. Only measures with `supports_precomputed_returns(r) == true` should implement this method.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`supports_precomputed_returns`](@ref)
"""
function expected_risk_from_returns end
"""
    no_bounds_risk_measure(r, args...; kwargs...)

Add a risk measure to a JuMP model without upper-bound constraints.

Generic function extended by concrete risk measure types.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
function no_bounds_risk_measure end
"""
    no_bounds_no_risk_expr_risk_measure(r, args...; kwargs...)

Add a risk measure to a JuMP model without upper-bound constraints and without adding a risk expression variable.

Generic function extended by concrete risk measure types.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
function no_bounds_no_risk_expr_risk_measure end
"""
    no_risk_expr_risk_measure(r, args...; kwargs...)

Return a copy of risk measure `r` with its risk-expression flag disabled while preserving its upper-bound constraint.

Generic function extended by concrete risk measure types. For hierarchical risk measures, returns `r` unchanged.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`no_bounds_risk_measure`](@ref)
  - [`bounds_risk_measure`](@ref)
"""
function no_risk_expr_risk_measure end
"""
    bounds_risk_measure(r, ub, args...; kwargs...)

Return a copy of risk measure `r` with its upper-bound constraint set to `ub`.

Generic function extended by concrete risk measure types. For hierarchical risk measures, returns `r` unchanged.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
  - [`no_bounds_risk_measure`](@ref)
  - [`no_bounds_no_risk_expr_risk_measure`](@ref)
"""
function bounds_risk_measure end

export Frontier, RiskMeasureSettings, HierarchicalRiskMeasureSettings, SumScalariser,
       MaxScalariser, MinScalariser, LogSumExpScalariser, expected_risk,
       expected_risk_from_returns, RiskMeasure, HierarchicalRiskMeasure, SquareRootBound,
       LinearBound, SquaredBound, SignificanceTailCalibration, SignificanceHeadCalibration,
       DeformationTailCalibration, DeformationHeadCalibration, ReturnsSeries,
       AbsoluteDrawdownSeries, RelativeDrawdownSeries, ScenarioCount, RateSignificance,
       EntropyBudget, HillTailDecay, RadialTailDecay, AmbiguityRadiusCalibration,
       AmbiguityTailWeightCalibration, ConcentrationRadius, RateRadius,
       DimensionalRateRadius, DualNormRadius, TailTermParity, NormCeilingCalibration,
       EffectiveAssetFloor
