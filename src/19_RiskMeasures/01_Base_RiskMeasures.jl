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

Used when the risk expression is in standard-deviation units but the user-supplied bound is in variance units (e.g. kurtosis and negative-skewness SOC formulations). This is also the transformation the default [`Variance`](@ref) formulation takes, so a `Variance` frontier is evenly spaced in **standard-deviation** units: on a 250x5 sample a five-point sweep gave standard deviations `0.004539514, 0.005969048, 0.007398587, 0.008828127, 0.010257666`, evenly spaced to `4.1e-9`, whose squares are the variances `2.0607e-5, 3.5630e-5, 5.4739e-5, 7.7936e-5, 1.0522e-4`, which are **not** evenly spaced.

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

Used when the risk expression and the user-supplied bound are already in the same units (e.g. SDP variance formulation where both sides are in variance units).

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

Used when the risk expression is in squared units but the user-supplied bound is in linear units (e.g. kurtosis SDP formulation).

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

Returns the risk-measure-specific solver if provided, otherwise falls back to the optimiser-level solver. Returns `nothing` if neither is available.

# Arguments

  - `risk_solvers`: Risk-measure-specific solver(s) or `nothing`.
  - `slv`: Optimiser-level solver(s) or `nothing`.

# Returns

  - Selected solver(s) or `nothing`.

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
    resolve_deferred_child(slot, pr::AbstractPriorResult)

Resolve one slot that [`deferred_slots`](@ref) declared, on behalf of the derived recursion in [`resolve_deferred_quantities`](@ref).

A slot holds one of three things, and one rule covers all three. A child measure resolves through its own method. A vector of children resolves element by element, which is the rule [`factory_child`](@ref) already applies on the other path. Anything else — a stated value, `nothing`, a **Deferred Quantity** the enclosing type resolves itself — is returned unchanged.

The vector arm is bounded by the element type rather than by `AbstractArray`, so a matrix slot is a value and never a container of children.

# Related

  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
  - [`factory_child`](@ref)
"""
function resolve_deferred_child(slot, pr::AbstractPriorResult)
    return resolve_deferred_quantities(slot, pr)
end
function resolve_deferred_child(slot::AbstractArray{<:Union{<:AbstractEstimator,
                                                            <:AbstractAlgorithm}},
                                pr::AbstractPriorResult)
    return [resolve_deferred_child(s, pr) for s in slot]
end
"""
    rebuild_with_slots(x, slots::NamedTuple)

Return a copy of `x` whose fields named by `slots` hold the values in `slots`.

The field list is derived from the type and the constructor is recovered from it, so nothing is written per type. The call is positional, so the inner constructor runs and every guard the type states is re-applied to the rebuilt value.

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

# Related

  - [`resolve_deferred_quantities`](@ref)
  - [`deferred_slots`](@ref)
  - [`DeferredQuantity`](@ref)
"""
function assert_declared_slot_resolver(x, slots::NamedTuple)
    for (key, slot) in pairs(slots)
        @argcheck(!isa(slot, DeferredQuantity),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Deferred Quantity, a `$(nameof(typeof(slot)))`, and `$(nameof(typeof(x)))` declares no `resolve_deferred_quantities` method to fit it. The derived recursion carries a child measure's own resolution, not a quantity of the enclosing type. Declare `resolve_deferred_quantities(x::$(nameof(typeof(x))), pr::AbstractPriorResult)` beside `deferred_slots`."))
    end
    return nothing
end
"""
    resolve_deferred_quantities(x, pr::AbstractPriorResult)

Resolve the children that [`deferred_slots`](@ref) declared, and return `x` itself when none of them changed.

This is the derived half of the resolution rule. A container declares its children once and both entry points follow: [`factory`](@ref) reaches them through [`@fprop`](@ref), and the `JuMP` builders reach them through this method. Neither needs a forwarding method per container.

A type that resolves a quantity of its own overrides this with its own method, which is more specific. So the derivation carries container recursion alone, and never guesses how a matrix, a tensor or the centre a moment was taken about comes out of a fit.

# Related

  - [`deferred_slots`](@ref)
  - [`resolve_deferred_child`](@ref)
  - [`assert_declared_slot_resolver`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function resolve_deferred_quantities(x, pr::AbstractPriorResult)
    slots = deferred_slots(x)
    if isempty(slots)
        return x
    end
    resolved = map(slot -> resolve_deferred_child(slot, pr), slots)
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

Computes a tail probability or a deformation parameter from the data a prior result carries, so that the quantity refits whenever the sample moves.

All concrete subtypes should subtype one of the two families under this root rather than the root itself. A plain number in place of a rule is the quantity itself, exactly as it is today.

A **Calibration Rule** is not a [`DeferredQuantity`](@ref), and the two mechanisms stay parallel end to end. A Deferred Quantity is *fitted* and the quantity is read off the fit; a rule fits nothing, and reads the sample size and the moments the prior result already carries. A rule also sees the effective observation weights, which [`resolve_slot`](@ref) does not carry. So a rule resolves through [`resolve_calibration_slot`](@ref), is declared through [`calibration_slots`](@ref), and is refused at a value-level entry point by [`assert_calibrated_slots`](@ref).

# Related

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`resolve_calibration_slot`](@ref)
  - [`DeferredQuantity`](@ref)
"""
abstract type AbstractCalibrationAlgorithm <: AbstractAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes a significance level, the tail probability that an `alpha` or a `beta` slot holds.

All concrete subtypes should subtype `AbstractSignificanceCalibrationAlgorithm`, and should be **callable**, because [`resolve_calibration_slot`](@ref) runs a rule by calling it. A plain `Function` of the same four arguments is therefore a rule as well, and needs no type at all. The family's two role types, [`SignificanceTailCalibration`](@ref) and [`SignificanceHeadCalibration`](@ref), subtype the family as well: a role names the end of the distribution the slot addresses and holds the rule in its `alg` field, and the same rule serves both ends.

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

All concrete subtypes should subtype `AbstractDeformationCalibrationAlgorithm`, and should be **callable**, on the same terms as the significance family. The family's two role types, [`DeformationTailCalibration`](@ref) and [`DeformationHeadCalibration`](@ref), subtype it as well.

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

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
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

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`SignificanceHeadCalibration`](@ref)
  - [`Num_SigTailCal`](@ref)
  - [`Func_SigCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct SignificanceTailCalibration <: AbstractSignificanceCalibrationAlgorithm
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

  - [`AbstractSignificanceCalibrationAlgorithm`](@ref)
  - [`SignificanceTailCalibration`](@ref)
  - [`Num_SigHeadCal`](@ref)
  - [`Func_SigCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct SignificanceHeadCalibration <: AbstractSignificanceCalibrationAlgorithm
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

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`DeformationHeadCalibration`](@ref)
  - [`Num_DefTailCal`](@ref)
  - [`Func_DefCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct DeformationTailCalibration <: AbstractDeformationCalibrationAlgorithm
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

  - [`AbstractDeformationCalibrationAlgorithm`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`Num_DefHeadCal`](@ref)
  - [`Func_DefCal`](@ref)
  - [`mirror_role`](@ref)
"""
@concrete struct DeformationHeadCalibration <: AbstractDeformationCalibrationAlgorithm
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

A rule gets no portfolio. A prior result carries no portfolio weight vector, so no rule can measure a portfolio's own loss series. It does get the solver: [`@propagatable`](@ref) runs the `@cprop` selection before the resolution, so a rule may call [`ERM`](@ref) or [`RRM`](@ref).

This is the parallel of [`resolve_slot`](@ref), and it is a second verb rather than a widening of the first for two reasons. `resolve_slot`'s body is `deferred_quantity(fit_deferred_quantity(dq, pr), key)`, a fit followed by an extraction, and a rule fits nothing. `resolve_slot` also carries neither `w` nor `slv`, which a rule needs. So the four role types stay **out** of the [`DeferredQuantity`](@ref) union.

The caller computes `w` itself, as `sel(r.w, pr.w)`, and threads it with the measure's own `slv`. A parent that carries no observation weights of its own passes `pr.w`, and one that carries no solver leaves `slv` at its default.

# Arguments

  - `slot`: The slot's occupant: a number, or one of the four role types.
  - `key`: Name of the slot that is being resolved.
  - `pr`: Prior result the rule reads.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver, or `nothing` when the measure carries none.

# Returns

  - `val::Number`: The calibrated quantity, or the stated value unchanged.

# Related

  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`calibration_slots`](@ref)
  - [`Func_SigCal`](@ref)
  - [`resolve_slot`](@ref)
"""
function resolve_calibration_slot(slot, ::Symbol, ::AbstractPriorResult, ::Any,
                                  ::Any = nothing)
    return slot
end
function resolve_calibration_slot(r::AbstractCalibrationAlgorithm, key::Symbol,
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

Refuse a **Calibration Rule** that reached a value-level entry point, which has no prior result to resolve it against.

[`expected_risk`](@ref) takes either a prior result or a plain returns matrix. Given the prior it resolves the measure through [`factory`](@ref) first. Given the matrix it cannot: a rule reads the sample size and the moments the prior carries, and it reads the effective observation weights, none of which a bare returns matrix supplies. So it refuses instead, and names the slot, the role standing in it and the way out.

This is the shape [`assert_resolved_slots`](@ref) already uses on the Deferred-Quantity side, and the message names both types with `nameof` for the same reason: a printed type carries a module prefix wherever the name is not visible from `Main`, and the message must read the same in every process.

The slots come from [`calibration_slots`](@ref) and the check recurses into whatever they hold, so a container is covered by its children's declarations.

# Related

  - [`calibration_slots`](@ref)
  - [`AbstractCalibrationAlgorithm`](@ref)
  - [`assert_resolved_slots`](@ref)
  - [`expected_risk`](@ref)
"""
function assert_calibrated_slots(x)
    for (key, slot) in pairs(calibration_slots(x))
        @argcheck(!isa(slot, AbstractCalibrationAlgorithm),
                  ArgumentError("`$(nameof(typeof(x))).$key` holds a Calibration Rule, a `$(nameof(typeof(slot)))`, and this entry point has no prior result to resolve it against. A rule reads the sample size, the moments and the effective observation weights, which a bare returns matrix does not carry. Pass the prior result itself — `expected_risk(r, w, pr, fees)` — or resolve the measure first with `factory(r, pr)`."))
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

A Range measure defaults its head slot to whatever its tail slot holds. A number crosses unchanged, and a tail role crosses as the head role of the same family holding the same `alg`, so the default survives the widening and no stated number moves.

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
  - ``T_{e}``: Effective sample size, which is Kish's when the observation weights are stated.
  - ``w_{i}``: Observation weight of period ``i``.

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
  - [`kappa_log`](@ref)
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
    bind_alpha(slot, alpha::Number)

Hand a resolved `alpha` to the rule that reads it, and return the slot's occupant with the number in place.

`alpha` and `kappa` are a **travelling pair**: [`EntropyBudget`](@ref) reads the significance level of a sibling slot, and [`resolve_calibration_slot`](@ref) carries a `Symbol` and no number. So the number travels through the rule itself. The slot owner's own resolution method resolves `alpha` first, calls this verb on the `kappa` slot, and resolves the result:

```julia
alpha = resolve_calibration_slot(x.alpha, :alpha, pr, w, slv)
kappa = resolve_calibration_slot(bind_alpha(x.kappa, alpha), :kappa, pr, w, slv)
```

The default is the identity, so a stated number, a plain function and a rule that reads no sibling all pass through untouched. A deformation role is rebuilt around the bound rule, which is what lets the verb take the slot rather than the rule the caller has to unwrap first. The significance family needs no method, because no significance rule reads a sibling and the identity is already the right answer for it.

# Arguments

  - `slot`: The slot's occupant: a number, a deformation role, or a rule.
  - `alpha`: The sibling slot's resolved significance level.

# Returns

  - `bound`: The occupant, with `alpha` in place wherever a rule reads it.

# Related

  - [`EntropyBudget`](@ref)
  - [`DeformationTailCalibration`](@ref)
  - [`DeformationHeadCalibration`](@ref)
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
"""
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the data a prior result carries, so that the radius refits whenever the sample moves.

An **Ambiguity Radius** is the radius of the ball of probability measures the model prices, and it is the coefficient of an un-squared norm penalty on the weight vector. It is neither a tail probability nor a deformation parameter, so it takes its own family beside the two of [`AbstractSignificanceCalibrationAlgorithm`](@ref) and [`AbstractDeformationCalibrationAlgorithm`](@ref), under the same root.

All concrete subtypes should subtype `AbstractAmbiguityRadiusCalibrationAlgorithm`, and should be **callable**, on the same terms as the two other families. The family's role type, [`AmbiguityRadiusCalibration`](@ref), subtypes it as well.

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
"""
abstract type AbstractAmbiguityRadiusCalibrationAlgorithm <: AbstractCalibrationAlgorithm end
"""
$(DocStringExtensions.TYPEDEF)

Computes an Esfahani-Kuhn tail weight, the weight that the `l` slot of a distributionally robust measure holds.

The tail weight scales the tail term of the Esfahani-Kuhn loss, and the mean term is not scaled by it. It is a different quantity from the radius, so it takes a different family and a different role: a radius rule placed in a tail-weight slot is refused at construction.

**No rule of this family ships.** The library computes no Esfahani-Kuhn tail weight, and inventing one would be a guess. The family exists so that a caller's own rule has a home, and [`Func_AmbTwtCal`](@ref) admits a plain function for exactly that case.

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
"""
abstract type AbstractAmbiguityTailWeightCalibrationAlgorithm <:
              AbstractCalibrationAlgorithm end
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

Field bound for the `alg` field of an ambiguity-tail-weight role: a rule of the family, or a plain function of the same four arguments. No rule of the family ships, so the plain function is the whole of the field's population today.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
const Func_AmbTwtCal = Union{<:Function, <:AbstractAmbiguityTailWeightCalibrationAlgorithm}
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

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`AmbiguityTailWeightCalibration`](@ref)
  - [`Num_AmbRadCal`](@ref)
  - [`Func_AmbRadCal`](@ref)
  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
"""
@concrete struct AmbiguityRadiusCalibration <: AbstractAmbiguityRadiusCalibrationAlgorithm
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

It is the counterpart of [`AmbiguityRadiusCalibration`](@ref), and carries the same shape. No rule of its family ships, so its `alg` holds a caller's own function today.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    AmbiguityTailWeightCalibration(;
        alg::Func_AmbTwtCal
    ) -> AmbiguityTailWeightCalibration

Keywords correspond to the struct's fields. `alg` has no default, because the rule is the whole content of the type.

# Related

  - [`AbstractAmbiguityTailWeightCalibrationAlgorithm`](@ref)
  - [`AmbiguityRadiusCalibration`](@ref)
  - [`Num_AmbTwtCal`](@ref)
  - [`Func_AmbTwtCal`](@ref)
"""
@concrete struct AmbiguityTailWeightCalibration <:
                 AbstractAmbiguityTailWeightCalibrationAlgorithm
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
$(DocStringExtensions.TYPEDEF)

Computes an ambiguity radius from the concentration of measure, so that the ball shrinks as the sample grows.

The radius is the Blanchet-Kang-Murthy form, a scale in the units of the returns times the square root of a chi-squared quantile over the sample size. The chi-squared factor is dimensionless and grows with the number of assets, so a wider universe buys a wider ball at a fixed confidence level, and a longer sample shrinks it.

`scale` states the units. A radius multiplies a norm of the weight vector, so it is in the units of the returns, and no caller can intuit that number from the confidence level alone. `scale = nothing` reads the average asset volatility off the prior result instead, which is the first pass the form's source recommends, and it moves with the sample the way the rest of the rule does.

`T` is the effective sample size when observation weights are stated, and the raw row count when they are not, on the same terms as [`ScenarioCount`](@ref). The radius prices estimation error, and the error of a weighted estimate falls with Kish's effective sample size rather than with the number of rows. [`RateRadius`](@ref) reads the raw row count instead, because a rate speaks of the length of the record.

The rule carries no range check of its own. It returns the quantity of the slot it stands in, so the slot owner's constructor is the whole validation, and a radius outside the slot's range is refused there, at fold time.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    ConcentrationRadius(;
        confidence::Number = 0.95,
        scale::Option{<:Number} = nothing
    ) -> ConcentrationRadius

Keywords correspond to the struct's fields. `confidence` defaults to `0.95`, and `scale` defaults to `nothing`, which reads the average asset volatility off the prior result.

## Validation

  - `0 < confidence < 1`.
  - If `scale` is not `nothing`: `scale > 0` and finite.

# Related

  - [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref)
  - [`RateRadius`](@ref)
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
    function ConcentrationRadius(confidence::Number, scale::Option{<:Number})
        assert_unit_interval(confidence, :confidence)
        assert_nonempty_gt0_finite_val(scale, :scale)
        return new{typeof(confidence), typeof(scale)}(confidence, scale)
    end
end
function ConcentrationRadius(; confidence::Number = 0.95, scale::Option{<:Number} = nothing)
    return ConcentrationRadius(confidence, scale)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the ambiguity radius that the concentration of measure gives on the sample that `pr` carries.

# Mathematical definition

```math
\\begin{align}
r &= s \\sqrt{\\dfrac{\\chi^{2}_{N,\\, q}}{T_{e}}}\\,,\\\\
s &= \\begin{cases}
\\dfrac{1}{N} \\sum\\limits_{i=1}^{N} \\sqrt{\\hat{\\mathbf{\\Sigma}}_{ii}} & \\textrm{if } \\texttt{scale} \\textrm{ is } \\texttt{nothing}\\\\
\\texttt{scale} & \\textrm{otherwise}
\\end{cases}\\,,\\\\
T_{e} &= \\begin{cases}
T & \\textrm{if } w \\textrm{ is } \\texttt{nothing}\\\\
\\dfrac{\\left(\\sum\\limits_{i=1}^{T} w_{i}\\right)^{2}}{\\sum\\limits_{i=1}^{T} w_{i}^{2}} & \\textrm{otherwise}
\\end{cases}\\,.
\\end{align}
```

Where:

  - ``r``: Ambiguity radius.
  - ``s``: Scale of the radius, in the units of the returns.
  - ``\\chi^{2}_{N,\\, q}``: Quantile of the chi-squared distribution with ``N`` degrees of freedom at confidence level ``q``.
  - $(math_dict[:N])
  - $(math_dict[:Sigma_hat_ii])
  - $(math_dict[:T])
  - ``T_{e}``: Effective sample size, which is Kish's when the observation weights are stated.
  - ``w_{i}``: Observation weight of period ``i``.

# Arguments

  - `alg`: The rule.
  - `key`: Name of the slot that is being resolved. The radius is the same for every key, so the two tails of a Range measure that carry one rule resolve to one number.
  - `pr`: Prior result the sample size, the asset count and the covariance matrix are read off.
  - `w`: Effective observation weights, or `nothing`.
  - `slv`: Effective solver. This rule needs none.

# Returns

  - `r::Number`: The ambiguity radius.

# Related

  - [`ConcentrationRadius`](@ref)
  - [`RateRadius`](@ref)
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::ConcentrationRadius)(::Symbol, pr::AbstractPriorResult, w, ::Any)
    N = size(pr.X, 2)
    T = isnothing(w) ? size(pr.X, 1) : sum(w)^2 / sum(abs2, w)
    scale = if isnothing(alg.scale)
        Statistics.mean(sqrt, LinearAlgebra.diag(pr.sigma))
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

  - ``r``: Ambiguity radius.
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
  - [`resolve_calibration_slot`](@ref)
"""
function (alg::RateRadius)(::Symbol, pr::AbstractPriorResult, ::Any, ::Any)
    return alg.c / sqrt(size(pr.X, 1))
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
sel(risk_variable::AbstractCalibrationAlgorithm, ::Any) = risk_variable
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
       DeformationTailCalibration, DeformationHeadCalibration, ScenarioCount,
       RateSignificance, EntropyBudget, AmbiguityRadiusCalibration,
       AmbiguityTailWeightCalibration, ConcentrationRadius, RateRadius
