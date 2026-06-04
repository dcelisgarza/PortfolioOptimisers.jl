"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all risk measure estimators in `PortfolioOptimisers.jl`.

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

Abstract supertype for all risk measure settings in `PortfolioOptimisers.jl`.

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
    function Frontier(N::Integer, factor::Number, bound::FrontierBoundEstimator)::Frontier
        @argcheck(N > zero(N))
        @argcheck(isfinite(factor))
        @argcheck(factor > zero(factor))
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

Settings type for configuring risk measure estimators in `PortfolioOptimisers.jl`.
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
        @argcheck(isfinite(scale))
        return new{typeof(scale), typeof(ub), typeof(rke)}(scale, ub, rke)
    end
end
function RiskMeasureSettings(; scale::Number = 1.0, ub::Option{<:RkRtBounds} = nothing,
                             rke::Bool = true)::RiskMeasureSettings
    return RiskMeasureSettings(scale, ub, rke)
end
"""
$(DocStringExtensions.TYPEDEF)

Settings type for configuring hierarchical risk measure estimators in `PortfolioOptimisers.jl`.

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
        @argcheck(isfinite(scale))
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
    risk_measure_view(rs, i, X)

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
function risk_measure_view(rs::AbstractBaseRiskMeasure, ::Any,
                           ::Any)::AbstractBaseRiskMeasure
    return rs
end
function risk_measure_view(rs::VecBaseRM, i, X::MatNum)
    return [risk_measure_view(r, i, X) for r in rs]
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
        @argcheck(gamma > zero(gamma))
        return new{typeof(gamma)}(gamma)
    end
end
function LogSumExpScalariser(; gamma::Number = 1.0)
    return LogSumExpScalariser(gamma)
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

  - [`risk_measure_view`](@ref)
"""
function risk_measure_nothing_scalar_array_view(::Nothing, ::Nothing, i)
    throw(ArgumentError("Both risk_variable and prior_variable are nothing."))
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
    throw(ArgumentError("Both risk_solver and prior_solver are nothing, cannot solve JuMP model."))
end
"""
    expected_risk(r, args...; kwargs...)

Compute the expected value of a risk measure.

Generic function extended by concrete risk measure types. Each method computes the risk value associated with its risk measure type, given a portfolio (or its return distribution).

# Related

  - [`AbstractBaseRiskMeasure`](@ref)
"""
function expected_risk end
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
       MaxScalariser, MinScalariser, LogSumExpScalariser, expected_risk, RiskMeasure,
       HierarchicalRiskMeasure, SquareRootBound, LinearBound, SquaredBound
