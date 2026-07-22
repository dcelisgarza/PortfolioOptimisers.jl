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

  - `N > 0`.
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
$(DocStringExtensions.TYPEDSIGNATURES)

Return a new vector of risk measures with [`factory`](@ref) applied element-wise.

# Related

  - [`VecBaseRM`](@ref)
  - [`factory`](@ref)
"""
function factory(rs::VecBaseRM, args...; kwargs...)
    return [factory(r, args...; kwargs...) for r in rs]
end
"""
    port_opt_view(rs, i, X)

Get a view or subset of a risk measure for asset cluster index `i`.

Returns the risk measure sliced for the given cluster or asset index. Used internally in hierarchical optimisation to apply risk measures to each cluster.

# Arguments

  - `rs`: Risk measure (or vector thereof).
  - `i`: Cluster or asset index.
  - `X`: Data matrix (used for dimension-aware slicing).

# Returns

  - Sliced risk measure or the original if no slicing is needed.

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
"""
function port_opt_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any,
                       args...)::AbstractBaseRiskMeasure
    return rs
end
function port_opt_view(rs::VecBaseRM, i, X::MatNum, args...)
    return [port_opt_view(r, i, X) for r in rs]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies used to combine multiple risk measures into a single scalar value for optimisation.

Subtype `Scalariser` to implement different methods for aggregating risk measures. These strategies are used in portfolio optimisation routines that require a single risk value from multiple risk measures.

# Related

  - [`NonHierarchicalScalariser`](@ref)
  - [`HierarchicalScalariser`](@ref)
"""
abstract type Scalariser <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies that combine multiple risk measures into a single scalar value compatible with all portfolio optimisation estimators.

Subtype `NonHierarchicalScalariser` to implement aggregation methods that work with all optimisation estimators.

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

Abstract supertype for scalarisation strategies that combine multiple risk measures into a single scalar value compatible only with hierarchical optimisations.

Subtype `HierarchicalScalariser` to implement aggregation methods that only work with hierarchical optimisation estimators.

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

Scalariser that selects the risk expression whose scaled value is the largest.

`MinScalariser` aggregates a vector of risk measures by selecting the minimum of their scaled values. The weights are specified in the `scale` field of [`RiskMeasureSettings`](@ref) or [`HierarchicalRiskMeasureSettings`](@ref). In clustering optimisations, the risk of each cluster is computed separately, so there is no coherence in which risk measure is chosen between clusters.

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
