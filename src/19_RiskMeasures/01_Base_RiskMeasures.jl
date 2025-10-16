abstract type AbstractBaseRiskMeasure <: AbstractEstimator end
abstract type NoOptimisationRiskMeasure <: AbstractBaseRiskMeasure end
abstract type OptimisationRiskMeasure <: AbstractBaseRiskMeasure end
abstract type RiskMeasure <: OptimisationRiskMeasure end
abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
abstract type AbstractRiskMeasureSettings <: AbstractEstimator end
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
struct RiskMeasureSettings{T1, T2, T3} <: AbstractRiskMeasureSettings
    scale::T1
    ub::T2
    rke::T3
    function RiskMeasureSettings(scale::Real,
                                 ub::Union{Nothing, <:Real, <:AbstractVector, <:Frontier},
                                 rke::Bool)
        if isa(ub, Real)
            @argcheck(isfinite(ub) && ub > zero(ub))
        elseif isa(ub, AbstractVector)
            @argcheck(!isempty(ub) && all(isfinite, ub) && all(x -> x > zero(x), ub))
        end
        @argcheck(isfinite(scale))
        return new{typeof(scale), typeof(ub), typeof(rke)}(scale, ub, rke)
    end
end
function RiskMeasureSettings(; scale::Real = 1.0,
                             ub::Union{Nothing, <:Real, <:AbstractVector, <:Frontier} = nothing,
                             rke::Bool = true)
    return RiskMeasureSettings(scale, ub, rke)
end
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
       MaxScalariser, LogSumExpScalariser, expected_risk
