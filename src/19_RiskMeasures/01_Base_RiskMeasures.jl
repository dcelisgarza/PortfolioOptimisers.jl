"""
    abstract type AbstractBaseRiskMeasure <: AbstractEstimator end

Abstract supertype for all risk measure estimators in PortfolioOptimisers.jl.

Defines the interface for risk measure types, which quantify portfolio risk using various statistical or econometric methods. All concrete risk measure types should subtype `AbstractBaseRiskMeasure` to ensure consistency and composability within the optimisation framework.

# Related Types

  - [`NoOptimisationRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type AbstractBaseRiskMeasure <: AbstractEstimator end
"""
    abstract type NoOptimisationRiskMeasure <: AbstractBaseRiskMeasure end

Abstract supertype for risk measures that are not intended for use in portfolio optimisation routines.

These risk measures are typically used for analysis, reporting, or diagnostics, and are not designed to be included as objectives or constraints in optimisation problems. Subtype this when implementing a risk measure that should not be selectable by optimisation algorithms.

# Related Types

  - [`AbstractBaseRiskMeasure`](@ref)
  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type NoOptimisationRiskMeasure <: AbstractBaseRiskMeasure end
"""
    abstract type OptimisationRiskMeasure <: AbstractBaseRiskMeasure end

Abstract supertype for risk measures that are intended for use in portfolio optimisation routines.

All concrete risk measures that can be used as objectives or constraints in optimisation problems should subtype `OptimisationRiskMeasure`. This ensures compatibility with the optimisation framework and enables composability with other estimators and algorithms.

# Related Types

  - [`RiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
  - [`NoOptimisationRiskMeasure`](@ref)
  - [`AbstractBaseRiskMeasure`](@ref)
"""
abstract type OptimisationRiskMeasure <: AbstractBaseRiskMeasure end
"""
    abstract type RiskMeasure <: OptimisationRiskMeasure end

Abstract supertype for standard risk measures used in portfolio optimisation.

Subtype `RiskMeasure` to implement concrete risk measures that quantify portfolio risk and can be used as objectives or constraints in optimisation problems. This type ensures compatibility with the optimisation framework and enables composability with other estimators and algorithms.

# Related Types

  - [`OptimisationRiskMeasure`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
"""
abstract type RiskMeasure <: OptimisationRiskMeasure end
"""
    abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end

Abstract supertype for hierarchical risk measures used in portfolio optimisation.

Subtype `HierarchicalRiskMeasure` to implement risk measures that operate on hierarchical or clustered portfolio structures. These measures are designed for use as objectives or constraints in optimisation problems that leverage asset clustering, hierarchical risk parity, or similar techniques.

# Related Types

  - [`OptimisationRiskMeasure`](@ref)
  - [`RiskMeasure`](@ref)
"""
abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
"""
    abstract type AbstractRiskMeasureSettings <: AbstractEstimator end

Abstract supertype for all risk measure settings in PortfolioOptimisers.jl.

Defines the interface for settings types that configure the behavior of risk measure estimators. All concrete risk measure settings types should subtype `AbstractRiskMeasureSettings` to ensure consistency and composability within the optimisation framework.

# Related Types

  - [`RiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
"""
abstract type AbstractRiskMeasureSettings <: AbstractEstimator end
"""
    struct Frontier{T1, T2, T3} <: AbstractAlgorithm
        N::T1
        factor::T2
        flag::T3
    end

Defines the number of points on the efficient frontier (Pareto Front).

# Fields

  - `N`: Number of points on the efficient frontier.
  - `factor`: Scaling factor, used to normalise moment-based risk measures.
  - `flag`: Boolean flag indicating whether to use the risk measure value as-is (`true`) or apply a square root (`false`).

# Constructor

Creates a `Frontier` with the specified number of points, scaling factor, and flag.

    Frontier(; N::Integer = 20)

Creates a `Frontier` with `N` points, a scaling factor of `1`, and `flag = true`. This is used to set the appropriate frontier bounds in [`variance_risk_bounds_val`](@ref) and [`second_moment_bound_val`](@ref).

    PortfolioOptimisers._Frontier(; N::Integer = 20, factor::Real, flag::Bool)

Keyword arguments correspond to the fields above.

## Validation

  - `N > 0`.
  - `isfinite(factor)` and `factor > 0`.

# Examples

```jldoctest
julia> Frontier(; N = 15)
Frontier
       N | Int64: 15
  factor | Int64: 1
    flag | Bool: true
```

# Related

  - [`RiskMeasureSettings`](@ref)
"""
struct Frontier{T1, T2, T3} <: AbstractAlgorithm
    N::T1
    factor::T2
    flag::T3
    function Frontier(N::Integer, factor::Real, flag::Bool)
        @argcheck(N > zero(N))
        @argcheck(isfinite(factor) && factor > zero(factor))
        return new{typeof(N), typeof(factor), typeof(flag)}(N, factor, flag)
    end
end
function Frontier(; N::Integer = 20)
    return Frontier(N, 1, true)
end
function _Frontier(; N::Integer = 20, factor::Real, flag::Bool)
    return Frontier(N, factor, flag)
end
"""
    struct RiskMeasureSettings{T1, T2, T3} <: AbstractRiskMeasureSettings
        scale::T1
        ub::T2
        rke::T3
    end

Settings type for configuring risk measure estimators in PortfolioOptimisers.jl.
Encapsulates scaling, upper bounds, and risk evaluation flags for risk measures used in optimisation routines.

# Fields

  - `scale`: Scaling factor applied to the risk measure.
  - `ub`: Upper bound(s) for the risk measure.
  - `rke`: Boolean flag indicating whether or not to include the risk measure in the `JuMP` model's risk expression. If `false`, a risk measure can be used to constrain the risk of an optimisation, without including it in the risk expression.

# Constructors

    RiskMeasureSettings(; scale::Real = 1.0,
                        ub::Union{Nothing, <:Real, <:AbstractVector, <:Frontier} = nothing,
                        rke::Bool = true)

Creates a `RiskMeasureSettings` instance with the specified scale, upper bound, and risk evaluation flag.

## Validation

  - `isfinite(scale)`.
  - `ub` is validated with [`assert_nonneg_finite_val`](@ref).

# Examples

```jldoctest
julia> RiskMeasureSettings(; scale = 2.0, ub = 0.5, rke = false)
RiskMeasureSettings{Float64, Float64, Bool}(2.0, 0.5, 0)
```

# Related

  - [`AbstractRiskMeasureSettings`](@ref)
  - [`RiskMeasure`](@ref)
  - [`Frontier`](@ref)
  - [`HierarchicalRiskMeasureSettings`](@ref)
"""
struct RiskMeasureSettings{T1, T2, T3} <: AbstractRiskMeasureSettings
    scale::T1
    ub::T2
    rke::T3
    function RiskMeasureSettings(scale::Real,
                                 ub::Union{Nothing, <:Real, <:AbstractVector, <:Frontier},
                                 rke::Bool)
        assert_nonneg_finite_val(ub)
        @argcheck(isfinite(scale))
        return new{typeof(scale), typeof(ub), typeof(rke)}(scale, ub, rke)
    end
end
function RiskMeasureSettings(; scale::Real = 1.0,
                             ub::Union{Nothing, <:Real, <:AbstractVector, <:Frontier} = nothing,
                             rke::Bool = true)
    return RiskMeasureSettings(scale, ub, rke)
end
"""
    struct HierarchicalRiskMeasureSettings{T1} <: AbstractRiskMeasureSettings
        scale::T1
    end

Settings type for configuring hierarchical risk measure estimators in PortfolioOptimisers.jl.

Used for `HierarchicalRiskMeasure`, where it is impossible to set a risk upper bound.

# Fields

  - `scale`: Scaling factor applied to the hierarchical risk measure.

# Constructors

```julia
HierarchicalRiskMeasureSettings(; scale::Real = 1.0)
```

Creates a `HierarchicalRiskMeasureSettings` instance with the specified scaling factor.

## Validation

  - `scale` must be finite.

# Examples

```jldoctest
julia> HierarchicalRiskMeasureSettings(; scale = 2.0)
HierarchicalRiskMeasureSettings{Float64}(2.0)
```

# Related

  - [`AbstractRiskMeasureSettings`](@ref)
  - [`HierarchicalRiskMeasure`](@ref)
  - [`RiskMeasureSettings`](@ref)
"""
struct HierarchicalRiskMeasureSettings{T1} <: AbstractRiskMeasureSettings
    scale::T1
    function HierarchicalRiskMeasureSettings(scale::Real)
        @argcheck(isfinite(scale))
        return new{typeof(scale)}(scale)
    end
end
function HierarchicalRiskMeasureSettings(; scale::Real = 1.0)
    return HierarchicalRiskMeasureSettings(scale)
end
function factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
    return rs
end
function factory(rs::AbstractVector{<:AbstractBaseRiskMeasure}, args...; kwargs...)
    return [factory(r, args...; kwargs...) for r in rs]
end
function risk_measure_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any)
    return rs
end
function risk_measure_view(rs::AbstractVector{<:AbstractBaseRiskMeasure}, i::AbstractVector,
                           X::AbstractMatrix)
    return [risk_measure_view(r, i, X) for r in rs]
end
abstract type Scalariser end
struct SumScalariser <: Scalariser end
struct MaxScalariser <: Scalariser end
struct LogSumExpScalariser{T1} <: Scalariser
    gamma::T1
    function LogSumExpScalariser(gamma::Real)
        @argcheck(gamma > zero(gamma))
        return new{typeof(gamma)}(gamma)
    end
end
function LogSumExpScalariser(; gamma::Real = 1.0)
    return LogSumExpScalariser(gamma)
end
function expected_risk end
function no_bounds_risk_measure end
function no_bounds_no_risk_expr_risk_measure end

export Frontier, RiskMeasureSettings, HierarchicalRiskMeasureSettings, SumScalariser,
       MaxScalariser, LogSumExpScalariser, expected_risk, RiskMeasure,
       HierarchicalRiskMeasure
