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

Defines the interface for settings types that configure the behavior of risk measure estimators. All concrete risk measure settings types should subtype `AbstractRiskMeasureSettings` to ensure consistency and composability within the optimisation framework.

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

Concrete subtypes control how numeric bound values are transformed before being applied to JuMP risk expressions. All subtypes should subtype `FrontierBoundEstimator`.

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

Used when the risk expression is in standard-deviation units but the user-supplied bound is in variance units (e.g. kurtosis and negative-skewness SOC formulations).

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

Defines the number of points on the efficient frontier (Pareto Front).

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

Construct a range of N evenly-spaced frontier parameter values.

Internal helper that generates a parameter grid (e.g., for risk bounds) used when sweeping the efficient frontier.

# Arguments

  - `N`: Number of frontier points (default 20).
  - `factor`: Scaling factor for the range.
  - `bound`: Controls whether to sweep from min-to-max or max-to-min.

# Returns

  - Vector of frontier parameter values.

# Related

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

Settings type for configuring risk measure estimators.
Encapsulates scaling, upper bounds, and risk evaluation flags for risk measures used in optimisation routines.

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

Settings type for configuring hierarchical risk measure estimators.

Used for `HierarchicalRiskMeasure`, where it is impossible to set a risk upper bound.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    HierarchicalRiskMeasureSettings(;
        scale::Number,
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

# Related

  - [`NonHierarchicalScalariser`](@ref)
  - [`HierarchicalScalariser`](@ref)
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

Scalariser that combines multiple risk measures using a weighted sum.

`SumScalariser` aggregates a vector of risk measures by computing the weighted sum of their scaled values. The weights are specified in the `scale` field of [`RiskMeasureSettings`](@ref) or [`HierarchicalRiskMeasureSettings`](@ref). This scalarisation strategy is used in portfolio optimisation routines that require a single risk value from multiple risk measures.

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
"""
struct SumScalariser <: NonHierarchicalScalariser end
"""
$(DocStringExtensions.TYPEDEF)

Scalariser that selects the risk expression whose scaled value is the largest.

`MaxScalariser` aggregates a vector of risk measures by selecting the maximum of their scaled values. The weights are specified in the `scale` field of [`RiskMeasureSettings`](@ref) or [`HierarchicalRiskMeasureSettings`](@ref). In clustering optimisations, the risk of each cluster is computed separately, so there is no coherence in which risk measure is chosen between clusters.

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
"""
struct MaxScalariser <: NonHierarchicalScalariser end
"""
$(DocStringExtensions.TYPEDEF)

Scalariser that selects the risk expression whose scaled value is the smallest.

`MinScalariser` aggregates a vector of risk measures by selecting the minimum of their scaled values. The weights are specified in the `scale` field of [`RiskMeasureSettings`](@ref) or [`HierarchicalRiskMeasureSettings`](@ref). In clustering optimisations, the risk of each cluster is computed separately, so there is no coherence in which risk measure is chosen between clusters.

It is the one [`HierarchicalScalariser`](@ref), because minimising a minimum is not convex and a `JuMP` optimiser cannot build it. That bound is the **model's**: at the value level the measures are already numbers, so `MinScalariser` is admitted by every `sca` keyword and every `sca` result field there, on equal footing with the other three.

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

Scalariser that aggregates multiple risk measures using the log-sum-exp function.

`LogSumExpScalariser` combines a vector of risk measures by applying the log-sum-exp transformation to their scaled values. The weights are specified in the `scale` field of [`RiskMeasureSettings`](@ref) or [`HierarchicalRiskMeasureSettings`](@ref).

The parameter `gamma` controls the approximation accuracy to the maximum function: as `gamma → 0`, the function approaches the weighted sum; as `gamma → ∞`, it approaches the maximum. This behaviour is only true in `JuMP`-based optimisations. In clustering optimisations, each cluster's risk is computed separately, so there is no coherence between clusters.

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
  - ``\\gamma``: Positive parameter controlling the interpolation between the weighted sum and the maximum functions.

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
  - everything else (moments) → [`nothing_scalar_array_selector`](@ref)

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
       LinearBound, SquaredBound
