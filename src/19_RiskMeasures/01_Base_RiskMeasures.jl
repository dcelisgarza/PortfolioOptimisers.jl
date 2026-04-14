"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all risk measure estimators in `PortfolioOptimisers.jl`.

Defines the interface for risk measure types, which quantify portfolio risk using various statistical or econometric methods. All concrete risk measure types should subtype `AbstractBaseRiskMeasure` to ensure consistency and composability within the optimisation framework.

All concrete risk measures can be used as functors (callable structs) to compute their associated risk quantity.

# Related Types

  - [`NonOptimisationRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type AbstractBaseRiskMeasure <: AbstractEstimator end
function needs_previous_weights(::AbstractBaseRiskMeasure)
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
function bigger_is_better(::AbstractBaseRiskMeasure)
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
function needs_previous_weights(r::VecBaseRM)
    return any(needs_previous_weights.(r))
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for risk measures that are not intended for use in portfolio optimisation routines.

These risk measures are typically used for analysis, reporting, or diagnostics, and are not designed to be included as objectives or constraints in optimisation problems. Subtype this when implementing a risk measure that should not be selectable by optimisation algorithms.

# Related Types

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

# Related Types

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

# Related Types

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

# Related Types

  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
"""
abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all risk measure settings in `PortfolioOptimisers.jl`.

Defines the interface for settings types that configure the behavior of risk measure estimators. All concrete risk measure settings types should subtype `AbstractRiskMeasureSettings` to ensure consistency and composability within the optimisation framework.

# Related Types

  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
"""
abstract type AbstractRiskMeasureSettings <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Defines the number of points on the efficient frontier (Pareto Front).

# Fields

  - `N`: Number of points on the efficient frontier.
  - `factor`: Scaling factor, used to normalise moment-based risk measures.
  - `flag`: Boolean flag indicating whether to use the risk measure value as-is (`true`) or apply a square root (`false`).

# Constructors

Creates a `Frontier` with the specified number of points, scaling factor, and flag.

    Frontier(;
        N::Integer = 20,
    ) -> Frontier

Creates a `Frontier` with `N` points, a scaling factor of `1`, and `flag = true`. This is used to set the appropriate frontier bounds in [`variance_risk_bounds_val`](@ref) and [`second_moment_bound_val`](@ref).

    PortfolioOptimisers._Frontier(;
        N::Integer = 20,
        factor::Number = 1.0,
        flag::Bool = true
    ) -> Frontier

Keywords correspond to the struct's fields.

## Validation

  - `N > 0`.
  - `isfinite(factor)` and `factor > 0`.

# Examples

```jldoctest
julia> Frontier(; N = 15)
Frontier
       N ┼ Int64: 15
  factor ┼ Int64: 1
    flag ┴ Bool: true
```

# Related

  - [`RiskMeasureSettings`](@ref)
"""
@concrete struct Frontier <: AbstractAlgorithm
    N
    factor
    flag
    function Frontier(N::Integer, factor::Number = 1, flag::Bool = true)
        @argcheck(N > zero(N))
        @argcheck(isfinite(factor))
        @argcheck(factor > zero(factor))
        return new{typeof(N), typeof(factor), typeof(flag)}(N, factor, flag)
    end
end
function Frontier(; N::Integer = 20)
    return Frontier(N, 1, true)
end
"""
    _Frontier(; N = 20, factor, flag)

Construct a range of N evenly-spaced frontier parameter values.

Internal helper that generates a parameter grid (e.g., for risk bounds) used when sweeping the efficient frontier.

# Arguments

  - `N`: Number of frontier points (default 20).
  - `factor`: Scaling factor for the range.
  - `flag`: Controls whether to sweep from min-to-max or max-to-min.

# Returns

  - Vector of frontier parameter values.

# Related

  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
"""
function _Frontier(; N::Integer = 20, factor::Number, flag::Bool)
    return Frontier(N, factor, flag)
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

  - `scale`: Scaling factor applied to the risk measure.
  - `ub`: Upper bound(s) for the risk measure.
  - `rke`: Boolean flag indicating whether or not to include the risk measure in the `JuMP` model's risk expression. If `false`, a risk measure can be used to constrain the risk of an optimisation, without including it in the risk expression.

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

  - [`AbstractRiskMeasureSettings`](@ref)
  - [`RiskMeasure`](@ref)
  - [`Frontier`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
"""
@concrete struct RiskMeasureSettings <: AbstractRiskMeasureSettings
    scale
    ub
    rke
    function RiskMeasureSettings(scale::Number, ub::Option{<:RkRtBounds}, rke::Bool)
        assert_nonempty_nonneg_finite_val(ub, :ub)
        @argcheck(isfinite(scale))
        return new{typeof(scale), typeof(ub), typeof(rke)}(scale, ub, rke)
    end
end
function RiskMeasureSettings(; scale::Number = 1.0, ub::Option{<:RkRtBounds} = nothing,
                             rke::Bool = true)
    return RiskMeasureSettings(scale, ub, rke)
end
"""
$(DocStringExtensions.TYPEDEF)

Settings type for configuring hierarchical risk measure estimators in `PortfolioOptimisers.jl`.

Used for `HierarchicalRiskMeasure`, where it is impossible to set a risk upper bound.

# Fields

  - `scale`: Scaling factor applied to the hierarchical risk measure.

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
    scale
    function HierarchicalRiskMeasureSettings(scale::Number)
        @argcheck(isfinite(scale))
        return new{typeof(scale)}(scale)
    end
end
function HierarchicalRiskMeasureSettings(; scale::Number = 1.0)
    return HierarchicalRiskMeasureSettings(scale)
end
function factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
    return rs
end
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
function risk_measure_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any)
    return rs
end
function risk_measure_view(rs::VecBaseRM, i, X::MatNum)
    return [risk_measure_view(r, i, X) for r in rs]
end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies used to combine multiple risk measures into a single scalar value for optimisation.

Subtype `Scalariser` to implement different methods for aggregating risk measures. These strategies are used in portfolio optimisation routines that require a single risk value from multiple risk measures.

# Related Types

  - [`NonHierarchicalScalariser`](@ref)
  - [`HierarchicalScalariser`](@ref)
"""
abstract type Scalariser <: AbstractEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for scalarisation strategies that combine multiple risk measures into a single scalar value compatible with all portfolio optimisation estimators.

Subtype `NonHierarchicalScalariser` to implement aggregation methods that work with all optimisation estimators.

# Related Types

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

# Related Types

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

  - `gamma`: Positive parameter controlling the interpolation between the weighted sum and the maximum functions.

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
function expected_risk end
function no_bounds_risk_measure end
function no_bounds_no_risk_expr_risk_measure end

export Frontier, RiskMeasureSettings, HierarchicalRiskMeasureSettings, SumScalariser,
       MaxScalariser, MinScalariser, LogSumExpScalariser, expected_risk, RiskMeasure,
       HierarchicalRiskMeasure
