abstract type AbstractMomentMeasureAlgorithm <: AbstractAlgorithm end
abstract type AbstractUnionLowOrderMomentMeasureAlgorithm <: AbstractMomentMeasureAlgorithm end
abstract type AbstractLowOrderMomentMeasureAlgorithm <:
              AbstractUnionLowOrderMomentMeasureAlgorithm end
abstract type AbstractLowOrderDeviationMeasureAlgorithm <:
              AbstractUnionLowOrderMomentMeasureAlgorithm end
function factory(alg::AbstractMomentMeasureAlgorithm, args...; kwargs...)
    return alg
end
struct FirstLowerMoment <: AbstractLowOrderMomentMeasureAlgorithm end
abstract type DeviationLowerMoment <: AbstractLowOrderMomentMeasureAlgorithm end
struct SecondLowerMoment{T1} <: DeviationLowerMoment
    alg::T1
    function SecondLowerMoment(alg::SecondMomentAlgorithm)
        return SecondLowerMoment(alg)
    end
end
function SecondLowerMoment(; alg::SecondMomentAlgorithm = SOCRiskExpr())
    return SecondLowerMoment(alg)
end
struct SecondCentralMoment{T1} <: DeviationLowerMoment
    alg::T1
    function SecondCentralMoment(alg::SecondMomentAlgorithm)
        return SecondCentralMoment(alg)
    end
end
function SecondCentralMoment(; alg::SecondMomentAlgorithm = SOCRiskExpr())
    return SecondCentralMoment(alg)
end
struct MeanAbsoluteDeviation <: DeviationLowerMoment end
struct LowOrderDeviation{T1, T2} <: AbstractLowOrderDeviationMeasureAlgorithm
    ve::T1
    alg::T2
    function LowOrderDeviation(ve::AbstractVarianceEstimator, alg::DeviationLowerMoment)
        @argcheck(!isa(alg, MeanAbsoluteDeviation))
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function LowOrderDeviation(; ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                           alg::DeviationLowerMoment = SecondLowerMoment())
    return LowOrderDeviation(ve, alg)
end
abstract type AbstractUnionHighOrderMomentMeasureAlgorithm <: AbstractMomentMeasureAlgorithm end
abstract type AbstractHighOrderMomentMeasureAlgorithm <:
              AbstractUnionHighOrderMomentMeasureAlgorithm end
abstract type AbstractHighOrderDeviationMeasureAlgorithm <:
              AbstractUnionHighOrderMomentMeasureAlgorithm end
struct ThirdLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthLowerMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct FourthCentralMoment <: AbstractHighOrderMomentMeasureAlgorithm end
struct HighOrderDeviation{T1, T2} <: AbstractHighOrderDeviationMeasureAlgorithm
    ve::T1
    alg::T2
    function HighOrderDeviation(ve::AbstractVarianceEstimator,
                                alg::AbstractHighOrderMomentMeasureAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function HighOrderDeviation(;
                            ve::AbstractVarianceEstimator = SimpleVariance(; me = nothing),
                            alg::AbstractHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return HighOrderDeviation(ve, alg)
end
for alg in (LowOrderDeviation, HighOrderDeviation)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights} = nothing)
                 return $(alg)(; ve = factory(alg.ve, w), alg = alg.alg)
             end
         end)
end
struct LowOrderMoment{T1, T2, T3, T4} <: AbstractMomentRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function LowOrderMoment(settings::RiskMeasureSettings,
                            w::Union{Nothing, <:AbstractWeights},
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                            alg::AbstractUnionLowOrderMomentMeasureAlgorithm)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function LowOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                        w::Union{Nothing, <:AbstractWeights} = nothing,
                        mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                        alg::AbstractUnionLowOrderMomentMeasureAlgorithm = FirstLowerMoment())
    return LowOrderMoment(settings, w, mu, alg)
end
struct HighOrderMoment{T1, T2, T3, T4} <: AbstractMomentHierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function HighOrderMoment(settings::RiskMeasureSettings,
                             w::Union{Nothing, <:AbstractWeights},
                             mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             alg::AbstractUnionHighOrderMomentMeasureAlgorithm)
        if isa(mu, AbstractVector)
            @argcheck(!isempty(mu) && all(isfinite, mu))
        elseif isa(mu, Real)
            @argcheck(isfinite(mu))
        end
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w), typeof(mu), typeof(alg)}(settings, w, mu,
                                                                         alg)
    end
end
function HighOrderMoment(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Union{Nothing, <:AbstractWeights} = nothing,
                         mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}} = nothing,
                         alg::AbstractUnionHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return HighOrderMoment(settings, w, mu, alg)
end
function calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                                    <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                            ::Any, x::AbstractVector)
    return mean(x)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing,
                                                       <:Any}}, ::Any, x::AbstractVector)
    return mean(x, r.w)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}}, w::AbstractVector, ::Any)
    return dot(w, r.mu)
end
function calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Real, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:Real, <:Any}}, ::Any,
                            ::Any)
    return r.mu
end
function calc_moment_val(r::Union{<:AbstractMomentRiskMeasure,
                                  <:AbstractMomentHierarchicalRiskMeasure,
                                  <:AbstractMomentNoOptimisationRiskMeasure},
                         w::AbstractVector, X::AbstractMatrix,
                         fees::Union{Nothing, <:Fees} = nothing)
    x = calc_net_returns(w, X, fees)
    target = calc_moment_target(r, w, x)
    return x .- target
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:FirstLowerMoment})(w::AbstractVector,
                                                                      X::AbstractMatrix,
                                                                      fees::Union{Nothing,
                                                                                  <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:LowOrderDeviation{<:Any, <:SecondLowerMoment{<:SqrtRiskExpr}}})(w::AbstractVector,
                                                                                              X::AbstractMatrix,
                                                                                              fees::Union{Nothing,
                                                                                                          <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:LowOrderDeviation{<:Any, <:SecondLowerMoment}})(w::AbstractVector,
                                                                              X::AbstractMatrix,
                                                                              fees::Union{Nothing,
                                                                                          <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:LowOrderDeviation{<:Any,
                                                <:SecondCentralMoment{<:SqrtRiskExpr}}})(w::AbstractVector,
                                                                                         X::AbstractMatrix,
                                                                                         fees::Union{Nothing,
                                                                                                     <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:LowOrderDeviation{<:Any, <:SecondCentralMoment}})(w::AbstractVector,
                                                                                X::AbstractMatrix,
                                                                                fees::Union{Nothing,
                                                                                            <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any, <:MeanAbsoluteDeviation})(w::AbstractVector,
                                                                           X::AbstractMatrix,
                                                                           fees::Union{Nothing,
                                                                                       <:Fees} = nothing)
    val = abs.(calc_moment_val(r, w, X, fees))
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment})(w::AbstractVector,
                                                                       X::AbstractMatrix,
                                                                       fees::Union{Nothing,
                                                                                   <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 3
    return isnothing(r.w) ? -mean(val) : -mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthLowerMoment})(w::AbstractVector,
                                                                        X::AbstractMatrix,
                                                                        fees::Union{Nothing,
                                                                                    <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any, <:FourthCentralMoment})(w::AbstractVector,
                                                                          X::AbstractMatrix,
                                                                          fees::Union{Nothing,
                                                                                      <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    val .= val .^ 4
    return isnothing(r.w) ? mean(val) : mean(val, r.w)
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:HighOrderDeviation{<:Any, <:ThirdLowerMoment}})(w::AbstractVector,
                                                                               X::AbstractMatrix,
                                                                               fees::Union{Nothing,
                                                                                           <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 3
    res = isnothing(r.w) ? -mean(val) : -mean(val, r.w)
    return res / (sigma * sqrt(sigma))
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:HighOrderDeviation{<:Any, <:FourthLowerMoment}})(w::AbstractVector,
                                                                                X::AbstractMatrix,
                                                                                fees::Union{Nothing,
                                                                                            <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
function (r::HighOrderMoment{<:Any, <:Any, <:Any,
                             <:HighOrderDeviation{<:Any, <:FourthCentralMoment}})(w::AbstractVector,
                                                                                  X::AbstractMatrix,
                                                                                  fees::Union{Nothing,
                                                                                              <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    sigma = Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
    val .= val .^ 4
    res = isnothing(r.w) ? mean(val) : mean(val, r.w)
    return res / sigma^2
end
for rt in (LowOrderMoment, HighOrderMoment)
    eval(quote
             function factory(r::$(rt), prior::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_factory(r.w, prior.w)
                 mu = nothing_scalar_array_factory(r.mu, prior.mu)
                 alg = factory(r.alg, w)
                 return $(rt)(; settings = r.settings, alg = alg, w = w, mu = mu)
             end
             function risk_measure_view(r::$(rt), i::AbstractVector, args...)
                 mu = nothing_scalar_array_view(r.mu, i)
                 return $(rt)(; settings = r.settings, alg = r.alg, w = r.w, mu = mu)
             end
         end)
end

export FirstLowerMoment, SecondLowerMoment, SecondCentralMoment, MeanAbsoluteDeviation,
       ThirdLowerMoment, FourthLowerMoment, FourthCentralMoment, LowOrderDeviation,
       HighOrderDeviation, LowOrderMoment, HighOrderMoment
