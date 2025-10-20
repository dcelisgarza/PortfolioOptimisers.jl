abstract type MomentMeasureAlgorithm <: AbstractAlgorithm end
abstract type LowOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
abstract type UnstandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end
abstract type StandardisedLowOrderMomentMeasureAlgorithm <: LowOrderMomentMeasureAlgorithm end
function factory(alg::MomentMeasureAlgorithm, args...; kwargs...)
    return alg
end
struct FirstLowerMoment <: UnstandardisedLowOrderMomentMeasureAlgorithm end
struct MeanAbsoluteDeviation <: UnstandardisedLowOrderMomentMeasureAlgorithm end
abstract type UnstandardisedSecondMomentAlgorithm <:
              UnstandardisedLowOrderMomentMeasureAlgorithm end
struct SecondLowerMoment{T1} <: UnstandardisedSecondMomentAlgorithm
    alg::T1
    function SecondLowerMoment(alg::SecondMomentFormulation)
        return new{typeof(alg)}(alg)
    end
end
function SecondLowerMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondLowerMoment(alg)
end
struct SecondCentralMoment{T1} <: UnstandardisedSecondMomentAlgorithm
    alg::T1
    function SecondCentralMoment(alg::SecondMomentFormulation)
        return new{typeof(alg)}(alg)
    end
end
function SecondCentralMoment(; alg::SecondMomentFormulation = SquaredSOCRiskExpr())
    return SecondCentralMoment(alg)
end
struct StandardisedLowOrderMoment{T1, T2} <: StandardisedLowOrderMomentMeasureAlgorithm
    ve::T1
    alg::T2
    function StandardisedLowOrderMoment(ve::AbstractVarianceEstimator,
                                        alg::UnstandardisedSecondMomentAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedLowOrderMoment(;
                                    ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                   me = nothing),
                                    alg::UnstandardisedSecondMomentAlgorithm = SecondLowerMoment())
    return StandardisedLowOrderMoment(ve, alg)
end
abstract type HighOrderMomentMeasureAlgorithm <: MomentMeasureAlgorithm end
abstract type UnstandardisedHighOrderMomentMeasureAlgorithm <:
              HighOrderMomentMeasureAlgorithm end
abstract type StandardisedHighOrderMomentMeasureAlgorithm <: HighOrderMomentMeasureAlgorithm end
struct ThirdLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
struct FourthLowerMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
struct FourthCentralMoment <: UnstandardisedHighOrderMomentMeasureAlgorithm end
struct StandardisedHighOrderMoment{T1, T2} <: StandardisedHighOrderMomentMeasureAlgorithm
    ve::T1
    alg::T2
    function StandardisedHighOrderMoment(ve::AbstractVarianceEstimator,
                                         alg::UnstandardisedHighOrderMomentMeasureAlgorithm)
        return new{typeof(ve), typeof(alg)}(ve, alg)
    end
end
function StandardisedHighOrderMoment(;
                                     ve::AbstractVarianceEstimator = SimpleVariance(;
                                                                                    me = nothing),
                                     alg::UnstandardisedHighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
    return StandardisedHighOrderMoment(ve, alg)
end
for alg in (StandardisedLowOrderMoment, StandardisedHighOrderMoment)
    eval(quote
             function factory(alg::$(alg), w::Union{Nothing, <:AbstractWeights} = nothing)
                 return $(alg)(; ve = factory(alg.ve, w), alg = alg.alg)
             end
         end)
end
struct LowOrderMoment{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function LowOrderMoment(settings::RiskMeasureSettings,
                            w::Union{Nothing, <:AbstractWeights},
                            mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                            alg::LowOrderMomentMeasureAlgorithm)
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
                        alg::LowOrderMomentMeasureAlgorithm = FirstLowerMoment())
    return LowOrderMoment(settings, w, mu, alg)
end
struct HighOrderMoment{T1, T2, T3, T4} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    mu::T3
    alg::T4
    function HighOrderMoment(settings::RiskMeasureSettings,
                             w::Union{Nothing, <:AbstractWeights},
                             mu::Union{Nothing, <:Real, <:AbstractVector{<:Real}},
                             alg::HighOrderMomentMeasureAlgorithm)
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
                         alg::HighOrderMomentMeasureAlgorithm = ThirdLowerMoment())
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
function calc_moment_val(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::AbstractVector,
                         X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
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
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondLowerMoment{<:SOCRiskExpr}}})(w::AbstractVector,
                                                                                               X::AbstractMatrix,
                                                                                               fees::Union{Nothing,
                                                                                                           <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any, <:SecondLowerMoment}})(w::AbstractVector,
                                                                                       X::AbstractMatrix,
                                                                                       fees::Union{Nothing,
                                                                                                   <:Fees} = nothing)
    val = min.(calc_moment_val(r, w, X, fees), zero(eltype(X)))
    return Statistics.var(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any,
                                                         <:SecondCentralMoment{<:SOCRiskExpr}}})(w::AbstractVector,
                                                                                                 X::AbstractMatrix,
                                                                                                 fees::Union{Nothing,
                                                                                                             <:Fees} = nothing)
    val = calc_moment_val(r, w, X, fees)
    return Statistics.std(r.alg.ve, val; mean = zero(eltype(val)))
end
function (r::LowOrderMoment{<:Any, <:Any, <:Any,
                            <:StandardisedLowOrderMoment{<:Any, <:SecondCentralMoment}})(w::AbstractVector,
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
                             <:StandardisedHighOrderMoment{<:Any, <:ThirdLowerMoment}})(w::AbstractVector,
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
                             <:StandardisedHighOrderMoment{<:Any, <:FourthLowerMoment}})(w::AbstractVector,
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
                             <:StandardisedHighOrderMoment{<:Any, <:FourthCentralMoment}})(w::AbstractVector,
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
       ThirdLowerMoment, FourthLowerMoment, FourthCentralMoment, StandardisedLowOrderMoment,
       StandardisedHighOrderMoment, LowOrderMoment, HighOrderMoment
